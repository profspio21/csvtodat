#!/usr/bin/env python3
"""Batch CSV -> DAT, preserving relative subfolders and filenames.

Requirements: Python 3.10+, numpy.
CSV rows = channels, columns = samples, after --skip-rows/--skip-cols.

Header: exactly 36 bytes, with big-endian fields:
    [6:14]  >Q time_start = Unix seconds from CSV filename
    [15:23] >d channel_interval = 0.815
    [23:27] >f time_interval = 2.0
    [28:32] >I channel_count = 500
Other header bytes are zero, or retained from --template-dat.
The CSV filename must contain YYYYMMDD-HHMMSSffffff and is preserved in the DAT
name. time_start is ASSUMED to mean Unix seconds, as in the previous version.
--utc-offset defaults to +09:00 (Japan); fractional seconds are discarded.
The >Q header field is an unsigned 64-bit integer, not a floating-point time.

Payload dtype is fixed to np.int16 (native byte order, little-endian on Windows
and standard x86 Linux). This is independent of the big-endian header.
Default payload layout is channel-sample (CSV orientation in C order).
Use --payload-layout sample-channel only if your reader requires a transpose.
Fractional values are truncated toward zero, without scaling or normalization.
Out-of-range values, NaN/Inf, and CSVs with a channel count other than 500 fail;
no clipping, padding, or cropping is performed.

Usage:
    python csv_to_dat.py input_folder output_folder --skip-rows 1 --skip-cols 1

Every CSV in the input folder and its subfolders is processed, including .CSV.
A nested output folder is excluded from discovery. Existing DATs are skipped.
Bad files are reported, and processing continues with the remaining files.
Exit status: 0 = batch completed, 1 = one or more files failed, 2 = setup error.
Files are verified before final rename; observers must handle move/rename events.
Use a single writer per output folder. Verification checks this binary format,
not any undocumented fields or reshape conventions in an external reader.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

import numpy as np

HEADER_SIZE = 36
CHANNEL_COUNT = 500
PAYLOAD_DTYPE = np.dtype(np.int16)
LAYOUTS = ("channel-sample", "sample-channel")


def read_csv_array(path, *, delimiter=",", skip_rows=0, skip_cols=0):
    """Read finite numeric [channels, samples], with explicit label skipping."""
    if skip_rows < 0 or skip_cols < 0:
        raise ValueError("skip_rows/skip_cols must be nonnegative")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))[skip_rows:]
    if not rows or any(len(row) <= skip_cols for row in rows):
        raise ValueError("CSV empty, blank row found, or skip_cols removes all data")
    rows = [row[skip_cols:] for row in rows]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("CSV rows have inconsistent sample counts")
    try:
        array = np.asarray(rows, dtype=np.float64)
    except ValueError as e:
        raise ValueError("Non-numeric CSV cell; check --skip-rows/--skip-cols") from e
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("CSV must be a nonempty finite 2D matrix (no NaN/Inf)")
    return array


def timestamp_to_uint64(csv_path, utc_offset="+09:00"):
    """Extract YYYYMMDD-HHMMSSffffff and return Unix seconds + discarded micros."""
    matches = re.findall(r"(?<!\d)(\d{8}-\d{12})(?!\d)", Path(csv_path).stem)
    if len(matches) != 1:
        raise ValueError("CSV filename must contain exactly one YYYYMMDD-HHMMSSffffff timestamp")
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", utc_offset)
    if not match or int(match[2]) > 23 or int(match[3]) > 59:
        raise ValueError("--utc-offset must look like +09:00 or +00:00")
    offset = timedelta(hours=int(match[2]), minutes=int(match[3]))
    if match[1] == "-":
        offset = -offset
    dt = datetime.strptime(matches[0], "%Y%m%d-%H%M%S%f").replace(tzinfo=timezone(offset))
    delta = dt.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = delta.days * 86400 + delta.seconds
    if not 0 <= seconds <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("time_start outside unsigned 64-bit range")
    return seconds, dt.microsecond


def build_header(*, time_start, template=None):
    """Write confirmed fields; remaining bytes are zero or supplied template bytes."""
    if not isinstance(time_start, int) or not 0 <= time_start <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("time_start must be an unsigned 64-bit integer")
    header = bytearray(HEADER_SIZE if template is None else template)
    if len(header) != HEADER_SIZE:
        raise ValueError("Template must contain exactly the first 36 header bytes")
    struct.pack_into(">Q", header, 6, time_start)
    struct.pack_into(">d", header, 15, 0.815)
    struct.pack_into(">f", header, 23, 2.0)
    struct.pack_into(">I", header, 28, CHANNEL_COUNT)
    return bytes(header)


def write_dat(array, output_path, *, time_start, payload_layout="channel-sample", template=None):
    """Write fixed np.int16 data and verify serialized bytes before final rename."""
    if payload_layout not in LAYOUTS:
        raise ValueError(f"payload_layout must be one of {LAYOUTS}")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError("Payload must be a finite, nonempty 2D array")
    if array.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"Expected {CHANNEL_COUNT} channels, got {array.shape[0]}")
    limits = np.iinfo(np.int16)
    if np.any(array < limits.min) or np.any(array > limits.max):
        raise ValueError("Values outside np.int16 range [-32768, 32767]; no clipping applied")
    oriented = array if payload_layout == "channel-sample" else array.T
    payload = np.ascontiguousarray(oriented, dtype=np.int16)
    fractional_count = int(np.count_nonzero(array != np.trunc(array)))
    header = build_header(time_start=time_start, template=template)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".dat":
        raise ValueError("Output filename must end in .dat")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix="csv_to_dat_", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header)
            f.write(payload.tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
        with tmp_path.open("rb") as f:
            if f.read(HEADER_SIZE) != header:
                raise ValueError("Header verification failed")
            recovered = np.fromfile(f, dtype=np.int16)
        if (recovered.size != payload.size
                or recovered.tobytes() != payload.tobytes(order="C")):
            raise ValueError("Payload verification failed")
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        os.rename(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dict(output=str(output_path.resolve()), channels=int(array.shape[0]),
                samples=int(array.shape[1]), payload_dtype="np.int16",
                payload_byte_order=sys.byteorder, payload_layout=payload_layout,
                fractional_values_truncated=fractional_count, header_bytes=HEADER_SIZE,
                file_bytes=HEADER_SIZE + payload.nbytes, channel_count=CHANNEL_COUNT,
                time_start=time_start, time_interval=2.0, channel_interval=0.815)


def convert_folder(input_folder, output_folder, *, skip_rows=0, skip_cols=0,
                   delimiter=",", payload_layout="channel-sample", template=None,
                   utc_offset="+09:00"):
    """Process each CSV independently, preserving its relative path in the output."""
    source = Path(input_folder).resolve()
    target = Path(output_folder).resolve()
    if not source.is_dir():
        raise ValueError(f"Input must be an existing folder: {source}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Output must be a folder: {target}")
    if skip_rows < 0 or skip_cols < 0:
        raise ValueError("--skip-rows/--skip-cols must be nonnegative")
    if len(delimiter) != 1:
        raise ValueError("--delimiter must be one character")
    if payload_layout not in LAYOUTS:
        raise ValueError(f"payload_layout must be one of {LAYOUTS}")
    build_header(time_start=0, template=template)  # Fail early for malformed templates.
    # Validate timezone once, rather than failing every file for a setup error.
    timestamp_to_uint64("19700102-000000000000.csv", utc_offset)
    exclude_target = target != source and target.is_relative_to(source)
    files = sorted(p for p in source.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".csv"
                   and (not exclude_target or not p.resolve().is_relative_to(target)))
    if not files:
        raise ValueError(f"No CSV files found under {source}")
    target.mkdir(parents=True, exist_ok=True)
    counts = dict(total=len(files), converted=0, skipped=0, failed=0)
    for src in files:
        dst = target / src.relative_to(source).with_suffix(".dat")
        if dst.exists():
            counts["skipped"] += 1
            print(json.dumps(dict(status="skipped", source=str(src), output=str(dst),
                                  reason="output already exists")))
            continue
        try:
            time_start, discarded_microseconds = timestamp_to_uint64(src, utc_offset)
            array = read_csv_array(src, delimiter=delimiter,
                                   skip_rows=skip_rows, skip_cols=skip_cols)
            report = write_dat(array, dst, time_start=time_start,
                               payload_layout=payload_layout, template=template)
            report["discarded_microseconds"] = discarded_microseconds
        except (OSError, ValueError, TypeError, csv.Error, struct.error, OverflowError) as e:
            counts["failed"] += 1
            print(json.dumps(dict(status="failed", source=str(src), error=str(e))))
            continue
        counts["converted"] += 1
        print(json.dumps(dict(status="converted", source=str(src), **report)))
    print(json.dumps(dict(status="summary", **counts)))
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_folder", type=Path)
    p.add_argument("output_folder", type=Path)
    p.add_argument("--payload-layout", choices=LAYOUTS, default="channel-sample")
    p.add_argument("--template-dat", type=Path)
    p.add_argument("--delimiter", default=",")
    p.add_argument("--utc-offset", default="+09:00", help="Filename timezone; default +09:00 (Japan)")
    p.add_argument("--skip-rows", type=int, default=0)
    p.add_argument("--skip-cols", type=int, default=0)
    args = p.parse_args(argv)
    try:
        template = None
        if args.template_dat is not None:
            with args.template_dat.open("rb") as f:
                template = f.read(HEADER_SIZE)
        counts = convert_folder(args.input_folder, args.output_folder,
                                skip_rows=args.skip_rows, skip_cols=args.skip_cols,
                                delimiter=args.delimiter, payload_layout=args.payload_layout,
                                template=template, utc_offset=args.utc_offset)
    except (OSError, ValueError, TypeError, csv.Error, struct.error) as e:
        p.exit(2, f"Error: {e}\n")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Read five consecutive CSVs as float64 arrays and plot their original values.

CSV rows are channels; columns are samples. Defaults skip exactly the first row
and first column. No normalization, filtering, clipping, integer conversion,
resampling, padding, or channel-count assumption is applied to the data.

Usage:
    python merge_csv_visualize.py folder_csv --skip-rows 1 --skip-cols 1
    python merge_csv_visualize.py folder_csv --start 5 --skip-rows 1 --skip-cols 1

The script reads one folder (not its subfolders). Filenames are sorted
lexicographically: YYYYMMDD-HHMMSSffffff.csv filenames therefore sort by time.
--start is a zero-based index into that sorted list. Exactly five files are
selected. Check their printed filenames represent the intended consecutive run.
No gaps are inserted for missing files; plot axes are sample/channel indices,
not physical time or distance. Rows must represent the same channels in the same
order in all files. All five CSVs must have the same number of channels.

Array output: [channels, total samples], saved as merged_START_END.npy.
Heatmap: original global min/max color scale, nearest-neighbor display, first
CSV row at the top. Dashed boundaries indicate each CSV, not detected vehicles.
Outputs go to folder_csv/csv_preview, or --output-dir. The plot opens unless
--no-show is passed. Existing outputs for the same group are replaced.

Dependencies: numpy, matplotlib. Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv_array(path, *, skip_rows=1, skip_cols=1, delimiter=","):
    """Read numeric cells as float64; preserve channel x sample orientation."""
    if skip_rows < 0 or skip_cols < 0:
        raise ValueError("skip_rows and skip_cols must be nonnegative")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))[skip_rows:]
    if not rows or any(len(row) <= skip_cols for row in rows):
        raise ValueError(f"{Path(path).name}: empty data or blank row")
    rows = [row[skip_cols:] for row in rows]
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{Path(path).name}: inconsistent sample counts between rows")
    try:
        array = np.asarray(rows, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"{Path(path).name}: nonnumeric data after skipping labels") from e
    if not np.isfinite(array).all():
        raise ValueError(f"{Path(path).name}: NaN or Inf found; data is not modified")
    return array


def merge_five_csv(folder, *, start=0, skip_rows=1, skip_cols=1, delimiter=","):
    """Return merged array, selected paths, and original sample counts."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Input must be an existing folder: {folder}")
    if start < 0:
        raise ValueError("start must be nonnegative")
    files = sorted((p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() == ".csv"), key=lambda p: p.name)
    selected = files[start:start + 5]
    if len(selected) != 5:
        raise ValueError(f"Need 5 CSVs from index {start}; found {len(selected)} (total {len(files)})")
    arrays = []
    for path in selected:
        array = load_csv_array(path, skip_rows=skip_rows, skip_cols=skip_cols,
                               delimiter=delimiter)
        if arrays and array.shape[0] != arrays[0].shape[0]:
            raise ValueError(f"{path.name}: {array.shape[0]} channels; expected {arrays[0].shape[0]}")
        print(f"{path.name}: shape={array.shape}, min={array.min():.9g}, max={array.max():.9g}")
        arrays.append(array)
    # Channels stay fixed; samples from each CSV follow the preceding CSV.
    merged = np.concatenate(arrays, axis=1)
    return merged, selected, [a.shape[1] for a in arrays]


def plot_array(array, *, sample_counts=None):
    """Visualize raw values using one shared color scale across the whole array."""
    array = np.asarray(array)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError("Expected a finite nonempty [channels, samples] array")
    if sample_counts is not None and (any(n <= 0 for n in sample_counts)
                                      or sum(sample_counts) != array.shape[1]):
        raise ValueError("sample_counts must be positive and match the total sample count")
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    image = ax.imshow(array, aspect="auto", origin="upper", interpolation="nearest",
                      cmap="viridis", vmin=float(array.min()), vmax=float(array.max()))
    ax.set_xlabel("Sample index (all CSVs combined)")
    ax.set_ylabel("Channel index (CSV row order)")
    ax.set_title(f"Original CSV values | {array.shape[0]} channels x {array.shape[1]} samples",
                 pad=35 if sample_counts is not None else 12)
    fig.colorbar(image, ax=ax, label="Original value")
    if sample_counts is not None:
        ends = np.cumsum(sample_counts)
        for boundary in ends[:-1]:
            ax.axvline(boundary - 0.5, color="white", linestyle="--", linewidth=0.8)
        centers = ends - np.asarray(sample_counts) / 2 - 0.5
        top = ax.secondary_xaxis("top")
        top.set_xticks(centers, [f"CSV {i + 1}" for i in range(len(sample_counts))])
        top.tick_params(length=0)
    return fig, ax


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--start", type=int, default=0, help="Index of first CSV, starting at 0")
    parser.add_argument("--skip-rows", type=int, default=1)
    parser.add_argument("--skip-cols", type=int, default=1)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening a window")
    args = parser.parse_args(argv)
    try:
        merged, selected, counts = merge_five_csv(
            args.folder, start=args.start, skip_rows=args.skip_rows,
            skip_cols=args.skip_cols, delimiter=args.delimiter)
        output = args.output_dir or args.folder / "csv_preview"
        output.mkdir(parents=True, exist_ok=True)
        stem = f"merged_{args.start:04d}_{args.start + 4:04d}"
        array_path = output / f"{stem}.npy"
        plot_path = output / f"{stem}.png"
        np.save(array_path, merged, allow_pickle=False)
        fig, _ = plot_array(merged, sample_counts=counts)
        fig.savefig(plot_path, dpi=180)
        print(f"Merged: shape={merged.shape}, dtype={merged.dtype}, min={merged.min():.9g}, max={merged.max():.9g}")
        print(f"Array: {array_path.resolve()}\nHeatmap: {plot_path.resolve()}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    except (OSError, ValueError, TypeError, csv.Error) as e:
        parser.exit(2, f"Error: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a copied Benchmark-3 variance table with selected components scaled."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def scale_entry(entry, component_indices, factor, n_rows):
    if entry[0] != "ABS":
        raise ValueError(f"Unsupported variance mode: {entry[0]!r}")
    if len(entry) == 2:
        values = np.asarray(entry[1], dtype=float).copy()
        if values.shape != (8,):
            raise ValueError(f"Expected eight variances, got {values.shape}")
        values[component_indices] *= factor
        return [entry[0], values.tolist()]
    if len(entry) == n_rows + 1:
        rows = []
        for values in entry[1:]:
            values = np.asarray(values, dtype=float).copy()
            if values.shape != (8,):
                raise ValueError(f"Expected eight variances, got {values.shape}")
            values[component_indices] *= factor
            rows.append(values.tolist())
        return [entry[0], *rows]
    raise ValueError(f"Unsupported variance entry with {len(entry)} items")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--factor", type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.factor) or args.factor <= 0:
        parser.error("factor must be positive and finite")

    names = ("USDP", "USDA", "UADP", "UADA", "UHRP", "UHRA", "UHAP", "UHAA")
    unknown = sorted(set(args.components) - set(names))
    if unknown:
        parser.error(f"unknown components: {unknown}")
    indices = [names.index(name) for name in args.components]
    frame = pd.read_pickle(args.input)
    for row in frame.index:
        for tool in frame.columns:
            frame.at[row, tool] = scale_entry(
                frame.at[row, tool], indices, args.factor, len(frame))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

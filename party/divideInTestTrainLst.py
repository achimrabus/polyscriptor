#!/usr/bin/env python3
"""
Split a Party dataset.arrow file into train/val sets (80/20 split).

Usage:
    python divideInTestTrainLst.py <dataset.arrow>

Example:
    python divideInTestTrainLst.py ./dataset.arrow

Outputs:
    train.arrow, val.arrow, train.lst, val.lst
"""

import sys
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python divideInTestTrainLst.py <dataset.arrow>")
        print("Example: python divideInTestTrainLst.py ./dataset.arrow")
        sys.exit(1)

    dataset_path = sys.argv[1]
    print(f"Reading {dataset_path}...")

    with pa.memory_map(dataset_path, 'rb') as source:
        table = ipc.open_file(source).read_all()

    num_rows = table.num_rows
    print(f"Total rows: {num_rows}")

    ids = np.random.permutation(num_rows)
    split = int(0.8 * num_rows)

    train_ids = ids[:split]
    val_ids = ids[split:]

    train_table = table.take(train_ids)
    val_table = table.take(val_ids)

    train_path = 'train.arrow'
    val_path = 'val.arrow'

    with pa.OSFile(train_path, 'wb') as sink:
        with ipc.new_file(sink, train_table.schema) as writer:
            writer.write(train_table)

    with pa.OSFile(val_path, 'wb') as sink:
        with ipc.new_file(sink, val_table.schema) as writer:
            writer.write(val_table)

    with open('train.lst', 'w', encoding="utf-8") as f:
        f.write(train_path + '\n')

    with open('val.lst', 'w', encoding="utf-8") as f:
        f.write(val_path + '\n')

    print(f"Done! Train: {len(train_ids)} rows, Val: {len(val_ids)} rows")


if __name__ == "__main__":
    main()

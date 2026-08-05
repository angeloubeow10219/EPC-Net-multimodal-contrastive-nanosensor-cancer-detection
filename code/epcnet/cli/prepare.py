import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epcnet.data.dataset import write_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="epc-prepare")
    parser.add_argument("--electrochemical", required=True)
    parser.add_argument("--plasmonic", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_matrix(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.load(source)
    return pd.read_csv(source, index_col=0).to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    required = {"label", "cohort", "instrument", "acquisition_date"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata columns missing: {sorted(missing)}")
    arrays = {
        "electrochemical": load_matrix(args.electrochemical),
        "plasmonic": load_matrix(args.plasmonic),
        "labels": metadata["label"].to_numpy(dtype=np.int64),
        "cohorts": metadata["cohort"].to_numpy(dtype=np.int64),
        "instruments": metadata["instrument"].to_numpy(dtype=np.int64),
        "acquisition_dates": metadata["acquisition_date"].to_numpy(dtype=np.int64),
    }
    write_hdf5(args.output, arrays)


if __name__ == "__main__":
    main()

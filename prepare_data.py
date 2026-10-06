"""Build every data folder used in the course from the UCI source file.

The dataset is not redistributed in this repository. Download it from the UCI
Machine Learning Repository, then run this script once from the repository root:

    1. Download: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    2. Unzip it and copy Concrete_Data.xls into this repository's root folder
    3. pip install xlrd            # pandas needs it to read .xls files
    4. python prepare_data.py

It reproduces the steps in 01_experimentation/1_data_ingest.ipynb (same splits,
same random_state) and writes:

    01_experimentation/exp_data/      original_data.csv, span1.csv, span2.csv,
                                      span-1/ and span-2/ with train/eval/test
    02_local_pipeline/loc_dev_data/   span1.csv, span-1/ with train/eval/test
    03_cloud_pipeline/bucket_data/data/
                                      span-1..3/ with train/, val/, test/
                                      (span-3 is a copy of span-2; spans 2 and 3
                                      get the _READY marker used in Module 9)

Source: Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5PK67 (CC BY 4.0).
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Short column names used throughout the course, in the UCI file's column order:
# cement, blast-furnace slag, fly ash, water, superplasticizer,
# coarse aggregate, fine aggregate, age (days), compressive strength (MPa)
COLUMNS = ["cement", "bfs", "fa", "water", "sp", "ca", "fa.1", "age", "c_str"]
SEED = 42
ROOT = Path(__file__).resolve().parent


def load_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {df.shape[1]} in {path}")
    df.columns = COLUMNS
    return df


def split_span(span: pd.DataFrame):
    """70% train, 15% test, 15% eval, exactly as in 1_data_ingest.ipynb."""
    train, temp = train_test_split(span, test_size=0.30, random_state=SEED)
    test, eval_ = train_test_split(temp, test_size=0.50, random_state=SEED)
    return train, test, eval_


def write_span(base: Path, train, test, eval_, eval_dir: str = "eval") -> None:
    for name, df, sub in (("train", train, "train"), ("test", test, "test"), ("eval", eval_, eval_dir)):
        out = base / sub
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(ROOT / "Concrete_Data.xls"),
                        help="Path to Concrete_Data.xls from UCI (default: repository root)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Source file not found: {src}\nDownload it first - see the instructions at the top of this script.")

    data = load_source(src)
    print(f"Loaded {len(data)} rows from {src.name}")

    # 1. Experimentation data (what 1_data_ingest.ipynb produces)
    exp = ROOT / "01_experimentation" / "exp_data"
    exp.mkdir(parents=True, exist_ok=True)
    data.to_csv(exp / "original_data.csv", index=False)

    span1, span2 = train_test_split(data, test_size=0.5, random_state=SEED)
    s1 = split_span(span1)
    s2 = split_span(span2)
    span1.to_csv(exp / "span1.csv", index=False)
    span2.to_csv(exp / "span2.csv", index=False)
    write_span(exp / "span-1", *s1)
    write_span(exp / "span-2", *s2)
    print("Span sizes (train, test, eval):", [len(x) for x in s1], [len(x) for x in s2])

    # 2. Local pipeline data (span 1 only)
    loc = ROOT / "02_local_pipeline" / "loc_dev_data"
    loc.mkdir(parents=True, exist_ok=True)
    span1.to_csv(loc / "span1.csv", index=False)
    write_span(loc / "span-1", *s1)

    # 3. Cloud bucket layout: gs://<bucket>/data/span-N/{train,val,test}/
    bucket = ROOT / "03_cloud_pipeline" / "bucket_data" / "data"
    write_span(bucket / "span-1", *s1, eval_dir="val")
    write_span(bucket / "span-2", *s2, eval_dir="val")
    span3 = bucket / "span-3"
    if span3.exists():
        shutil.rmtree(span3)
    shutil.copytree(bucket / "span-2", span3)
    for n in (2, 3):
        (bucket / f"span-{n}" / "_READY").touch()

    print("Done. Data written to 01_experimentation/exp_data, 02_local_pipeline/loc_dev_data "
          "and 03_cloud_pipeline/bucket_data/data")


if __name__ == "__main__":
    main()

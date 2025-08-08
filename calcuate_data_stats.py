import os
import json
import polars as pl

if __name__ == "__main__":
    data_fpath = "data/EHR_QA/MIMICIV/icu_mortality/nl/all.parquet"
    df = pl.read_parquet(data_fpath)

    # 1) Extract the column as a Python list
    raw = df["meta_data"].to_list()

    # 2) If entries are JSON strings, parse them; otherwise assume they're already dicts
    records = []
    for rec in raw:
        if isinstance(rec, str):
            records.append(json.loads(rec))
        else:
            records.append(rec)

    # 3) Build a new Polars DataFrame from the list of dicts
    meta_df = pl.DataFrame(records)

    # 4) Count unique subject_id
    num_subjects = len(meta_df["subject_id"].unique())
    print(f"Number of unique subjects: {num_subjects}")

    # 5) Count unique icustay_id
    num_icustays = len(meta_df["icustay_id"].unique())
    print(f"Number of unique icustays: {num_icustays}")

from dataclasses import dataclass


@dataclass
class MIMICICUMortalityDatasetConfig:
    train_data_path = 'data/EHR_QA/MIMICIV/icu_mortality/nl/train.parquet'
    tuning_data_path = 'data/EHR_QA/MIMICIV/icu_mortality/nl/tuning.parquet'
    held_out_data_path = 'data/EHR_QA/MIMICIV/icu_mortality/nl/held_out.parquet'

    input_max_length = 32*1024
    num_classes = 2


@dataclass
class RegularMIMICConfig:
    data_dir = "data/EHR_regular"
    time_resolution = "1h"
    task = "icu_phenotyping"
    num_classes = 2

    demo_csv_name = 'demo.parquet'
    ts_csv_name = 'ts.parquet'
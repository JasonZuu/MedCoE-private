import torch
from torch.utils.data import Dataset
import polars as pl
from pathlib import Path
import numpy as np

from config.data_config import RegularMIMICConfig
from dataset.map_fn import map_gender_to_index, map_race_to_index
from utils.constants import phenotypes_ccs 


class RegularSampleDataset(Dataset):
    """
    Randomly sampling from all records from included sets
    """
    def __init__(self, config: RegularMIMICConfig,
                 set_name: str):
        super().__init__()
        # load data
        self.data_dir = Path(config.data_dir) / set_name
        self.demo_df = pl.read_parquet(self.data_dir / config.demo_csv_name).unique()
        self.ts_df = pl.read_parquet(self.data_dir / config.ts_csv_name).unique()
        
        self.set_name = set_name
        self.config = config
        self.num_classes = config.num_classes
        self.task = config.task
        if self.task == "icu_phenotyping":
            self.phenotype_list = list(phenotypes_ccs.keys())
        self.demo_dims = self.demo_df.width - 4 # remove subject_id, hadm_id, icustay_id, label
        self.ts_dims = self.ts_df.width - 4 # remove subject_id, hadm_id, icustay_id, label

    def __len__(self):
        return self.demo_df.height

    def __getitem__(self, idx):
        # get demo data
        demo_row = self.demo_df.row(idx, named=True)
        subject_id = demo_row['subject_id']
        hadm_id = demo_row['hadm_id']
        icustay_id = demo_row['icustay_id']
        if self.task == "icu_phenotyping":
            phenotypes = demo_row['label']
            label = self.parse_phenotyping_label(phenotypes)
        else:
            label = demo_row['label']
            label = self.transform_onehot(label)
        gender = map_gender_to_index(demo_row["GENDER"])
        age = demo_row["AGE"]
        race = map_race_to_index(demo_row["RACE"])
        demo = [gender, int(age), race]
        
        # get ts data
        ts_row = (
            self.ts_df
            .filter(
                (pl.col('subject_id') == subject_id) &
                (pl.col('hadm_id') == hadm_id) &
                (pl.col('icustay_id') == icustay_id)
            ))
        ts = ts_row.drop(['subject_id', 'hadm_id', 'icustay_id', "timestep"]).to_numpy()

        return {
            'demo': torch.tensor(demo, dtype=torch.float),
            'ts': torch.tensor(ts, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }
    
    def parse_phenotyping_label(self, phenotypes: list):
        """
        Parse the phenotyping label from the dataset.
        """
        label_onehot = np.zeros(self.num_classes)
        for phenotype in phenotypes:
            if phenotype in self.phenotype_list:
                label_onehot[self.phenotype_list.index(phenotype)] = 1
        return label_onehot

    def transform_onehot(self, label: int):
        """
        Parse the mortality label from the dataset.
        """
        label_onehot = np.zeros(self.num_classes)
        label_onehot[label] = 1
        return label_onehot

if __name__ == "__main__":
    config = RegularMIMICConfig()
    dataset = RegularSampleDataset(config, set_name='train')
    print(dataset[0])

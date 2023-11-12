import pandas as pd
import numpy as np

def load_hr_spo2_data(FILE_PATH):
    return pd.read_csv(FILE_PATH)

def load_acc_temp_eda_data(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    df = df.dropna().reset_index(drop=True)
    return df
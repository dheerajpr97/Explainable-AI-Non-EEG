import numpy as np
import os

LABELS = ['Relax_1', 'Relax_2', 'Relax_3', 'Relax_4', 'PhysicalStress', 'CognitiveStress', 'EmotionalStress']
LABELS_DICT = {
    'Relax' : 0,
    'PhysicalStress': 1,
    'CognitiveStress': 2,
    'EmotionalStress': 3
}
SUBJECTS = ['Subject_1', 'Subject_10', 'Subject_11', 'Subject_12', 'Subject_13', 'Subject_14', 'Subject_15', 'Subject_16', 'Subject_17', 'Subject_18',
             'Subject_19', 'Subject_2', 'Subject_20', 'Subject_3', 'Subject_4', 'Subject_5', 'Subject_6', 'Subject_7', 'Subject_8', 'Subject_9']
MODALS = ['HeartRate', 'SpO2', 'Temp', 'EDA', 'AccX', 'AccY', 'AccZ']
MODALS_1 = ['HeartRate', 'SpO2']
MODALS_2 = ['Temp', 'EDA', 'AccX', 'AccY', 'AccZ']

ROOT_DIR = os.path.join(os.getcwd())
DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'data')

HR_SPO2_FILE_PATH = os.path.join(DATA_ROOT_DIR,'dataset', 'Subjects_all_HR_SpO2.csv')
ACC_TEMP_EDA_FILE_PATH = os.path.join(DATA_ROOT_DIR,'dataset', 'Subjects_all_AccTempEDA.csv')

SAVED_DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'created_dataset')
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, 'saved_models')

TRAIN_DATA_MEAN = 29.475756
TRAIN_DATA_STD = 38.412533
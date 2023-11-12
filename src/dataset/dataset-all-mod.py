import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from src.utils.utils import load_hr_spo2_data, load_acc_temp_eda_data
from src.constants.constants import LABELS, SUBJECTS, MODALS, MODALS_1, MODALS_2

df_hr_spo2 = load_hr_spo2_data('data/dataset/Subjects_all_HR_SpO2.csv')
df_acc_temp_eda = load_acc_temp_eda_data('data/dataset/Subjects_all_AccTempEDA.csv')

class DataframePrepAllMod:
    def __init__(self, subjects, modals, modals_1, modals_2, df_hr_spo2, df_acc_temp_eda, label, frame_rate=60, max_size=360):
        """
        Initializes the DataframePrepAllMod class.

        Parameters:
        - subjects (list): List of subjects.
        - modals (list): List of all modalities.
        - modals_1 (list): List of modalities for the first dataset.
        - modals_2 (list): List of modalities for the second dataset.
        - df_hr_spo2 (pd.DataFrame): DataFrame for Heartrate and SpO2 data.
        - df_acc_temp_eda (pd.DataFrame): DataFrame for accelerometer, temperature, and EDA data.
        - label (list): List of labels.
        - frame_rate (int): Length of each segment (default: 60).
        - max_size (int): Duration to which the overall duration has to be up/downsampled (default: 360).
        """
        self.subjects = subjects
        self.modals = modals
        self.modals_1 = modals_1
        self.modals_2 = modals_2
        self.df_hr_spo2 = df_hr_spo2
        self.df_acc_temp_eda = df_acc_temp_eda
        self.label = label
        self.frame_rate = frame_rate
        self.max_size = max_size

    def dataframe_prep_allmod(self):
        """
        Up/down samples raw data to 'max_size' and chunks to fixed segments of 'frame_rate' each for all modalities.

        Returns:
        - pd.DataFrame: Processed DataFrame with up/downsampled and segmented data for all modalities.
        """
        df_mod_all = self.process_modalities()

        # Combine modalities into a single DataFrame
        data = self.combine_modalities(df_mod_all)

        # Create the final DataFrame
        df_fin = self.create_final_dataframe(data)

        return df_fin

    def process_modalities(self):
        df_mod_all = pd.DataFrame()

        for k in range(len(self.modals)):
            df = self.df_hr_spo2 if self.modals[k] in self.modals_1 else self.df_acc_temp_eda

            for j in range(len(self.subjects)):
                for i in range(len(self.label)):
                    df_list = df[(df['Subject'] == self.subjects[j]) & (df['Label'] == self.label[i])].reset_index(drop=True)
                    arr_list = np.array(df_list[self.modals[k]])

                    arr_interp = interp1d(np.arange(arr_list.size), arr_list)
                    arr_reshape = np.reshape(arr_interp(np.linspace(0, arr_list.size-1, int(self.max_size))),
                                             (int(len(arr_interp(np.linspace(0, arr_list.size-1, int(self.max_size)))) / self.frame_rate), self.frame_rate))

                    df_mod_one = pd.DataFrame(arr_reshape)

                    df_mod_one['Label'] = self.label_to_numeric(self.label[i])
                    df_mod_one['Label_ori'] = self.label[i]

                    df_mod_one['Subject'] = self.subjects[j]
                    df_mod_one['Modality'] = self.modals[k]
                    df_mod_all = pd.concat([df_mod_all, df_mod_one])

        return df_mod_all

    def label_to_numeric(self, label):
        # Define a mapping from label prefixes to numeric values
        label_prefixes = {
            'Relax': 0,
            'PhysicalStress': 1,
            'CognitiveStress': 2,
            'EmotionalStress': 3,
            # Add more prefixes as needed
        }

        # Extract the prefix from the label
        label_prefix = next((prefix for prefix in label_prefixes if label.startswith(prefix)), None)

        # Convert labels to numeric values using the mapping
        return label_prefixes.get(label_prefix, -1)  # Default to -1 for unknown labels (customize as needed)


    def combine_modalities(self, df_mod_all):
        data = df_mod_all[df_mod_all['Modality'] == self.modals[0]].iloc[:, :-1].values

        for i in range(1, len(self.modals)):
            data = np.dstack((data, (df_mod_all[df_mod_all['Modality'] == self.modals[i]].iloc[:, :-1].values)))

        return data

    def create_final_dataframe(self, data):
        df_fin = pd.DataFrame()
        df_fin['Data'] = list(data)
        df_fin['Subject'] = list(data[:, -1, 0])
        df_fin['Label'] = list(data[:, -3, 0])
        df_fin['Label_ori'] = list(data[:, -2, 0])

        return df_fin

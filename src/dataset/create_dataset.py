import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.constants.constants import (LABELS, MODALS, MODALS_1, MODALS_2,
                                     SUBJECTS)
from src.utils.utils import load_csv_to_dataframe


class DataframePrepAllMod:
    def __init__(self, subjects, modals, modals_1, modals_2, hr_spo2_file, acc_temp_eda_file, label, frame_rate=60, max_size=360):
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
        # Store the input arguments as instance variables
        self.subjects = subjects
        self.modals = modals
        self.modals_1 = modals_1
        self.modals_2 = modals_2
        self.hr_spo2_file = hr_spo2_file
        self.acc_temp_eda_file = acc_temp_eda_file
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
        """
        Process modalities and return a concatenated dataframe of processed data.
        """

        # Create an empty dataframe to store the processed data
        df_mod_all = pd.DataFrame()

        # Load HR and SpO2 data from csv file
        df_hr_spo2 = load_csv_to_dataframe(self.hr_spo2_file)

        # Load accelerometer, temperature, and EDA data from csv file
        df_acc_temp_eda = load_csv_to_dataframe(self.acc_temp_eda_file)
        
        # Iterate over each modality
        for modality in self.modals:
            # Select the dataframe based on the modality
            df = df_hr_spo2 if modality in self.modals_1 else df_acc_temp_eda

            # Process subjects' labels for the selected modality
            processed_data = self.process_subjects_labels(df, modality)

            # Concatenate the processed data to the overall dataframe
            df_mod_all = pd.concat([df_mod_all, processed_data])

        return df_mod_all

    def process_subjects_labels(self, df, modality):
        """
        Process the subjects and labels in the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            modality (str): The modality to be processed.

        Returns:
            pandas.DataFrame: The processed data.
        """
        processed_data_list = []

        # Iterate over each subject
        for subject in self.subjects:
            # Iterate over each label
            for label in self.label:
                # Filter the DataFrame for the current subject and label
                df_list = df.loc[(df['Subject'] == subject) & (df['Label'] == label), :].reset_index(drop=True)
                arr_list = np.array(df_list[modality])

                # Interpolate the array to match the desired size
                arr_interp = interp1d(np.arange(arr_list.size), arr_list)
                arr_reshape = np.reshape(arr_interp(np.linspace(0, arr_list.size - 1, int(self.max_size))),
                                        (int(len(arr_interp(np.linspace(0, arr_list.size - 1, int(self.max_size)))) /
                                            self.frame_rate), self.frame_rate))

                # Create a DataFrame from the reshaped array
                df_mod_one = pd.DataFrame(arr_reshape)

                # Add additional columns to the DataFrame
                df_mod_one['Label'] = self.label_to_numeric(label)
                df_mod_one['Label_ori'] = label
                df_mod_one['Subject'] = subject
                df_mod_one['Modality'] = modality

                # Append the processed DataFrame to the list
                processed_data_list.append(df_mod_one)

        # Concatenate all processed DataFrames into a single DataFrame
        return pd.concat(processed_data_list)

    def label_to_numeric(self, label):
        # Define a mapping from label prefixes to numeric values
        label_prefixes = {
            'Relax': 0,
            'PhysicalStress': 1,
            'CognitiveStress': 2,
            'EmotionalStress': 3,
    }

        # Extract the prefix from the label
        label_prefix = next((prefix for prefix in label_prefixes if label.startswith(prefix)), None)

        # Convert labels to numeric values using the mapping
        return label_prefixes.get(label_prefix, -1)  # Default to -1 for unknown labels


    def combine_modalities(self, df_mod_all):
        """
        Combine modalities from a dataframe.
        
        Args:
            df_mod_all (pandas.DataFrame): Dataframe containing modalities.
            
        Returns:
            numpy.ndarray: Combined data from all modalities.
        """
        # Extract data for the first modality
        data = df_mod_all[df_mod_all['Modality'] == self.modals[0]].iloc[:, :-1].values

        # Loop through the remaining modalities and stack the data
        for i in range(1, len(self.modals)):
            data = np.dstack((data, (df_mod_all[df_mod_all['Modality'] == self.modals[i]].iloc[:, :-1].values)))

        return data

    def create_final_dataframe(self, data):
        """
        Create a final dataframe from the given data.

        Parameters:
        - data: numpy array, the input data

        Returns:
        - df_fin: pandas DataFrame, the final dataframe
        """

        # Create an empty dataframe
        df_fin = pd.DataFrame()

        # Add 'Data' column to the dataframe
        df_fin['Data'] = list(data)

        # Add 'Subject' column to the dataframe
        df_fin['Subject'] = list(data[:, -1, 0])

        # Add 'Label' column to the dataframe
        df_fin['Label'] = list(data[:, -3, 0])

        # Add 'Label_ori' column to the dataframe
        df_fin['Label_ori'] = list(data[:, -2, 0])

        return df_fin

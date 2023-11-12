import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class DataframePrepHR:
    def __init__(self, subjects, label):
        """
        Initializes the DataframePrepOneMod class.

        Parameters:
        - subjects (list): List of subjects.
        - label (list): List of labels.
        """
        self.subjects = subjects
        self.label = label
        
    def interpolate_and_reshape(self, arr_list, max_size, frame_rate):
        """
        Interpolates and reshapes the input array.

        Parameters:
        - arr_list (list): List of input arrays.
        - max_size (int): Maximum size of the output array.

        Returns:
        - np.ndarray: Reshaped and interpolated array.
        """
        # Interpolate the input array 
        arr_interp = interp1d(np.arange(arr_list.size), arr_list)

        # Reshape the interpolated array
        arr_reshape = np.reshape(arr_interp(np.linspace(0, arr_list.size-1, max_size)),
                         (int(len(arr_interp(np.linspace(0, arr_list.size-1, max_size)))/frame_rate), frame_rate))

        return arr_reshape


    def dataframe_prep_hr(self, df, frame_rate=60, max_size=360):
        """
        Up/down samples raw data to 'max_size' and chunks to fixed segments of 'frame_rate' each for one modality.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing raw data.
        - frame_rate (int): Length of each segment (default: 60).
        - max_size (int): Duration to which the overall duration has to be up/downsampled (default: 360).

        Returns:
        - pd.DataFrame: Processed DataFrame with up/downsampled and segmented data.
        """
        # Declare empty lists and dataframes to store processed data
        df_list = {}
        df_final = pd.DataFrame()

        # Loop through all the subjects
        for j in range(len(self.subjects)):
            # Loop through all the labels
            for i in range(len(self.label)):
                var = f'df_{self.label[i]}_{self.subjects[j]}'  # Variable name based on label and subject
                df_list[var] = df[(df['Subject'] == self.subjects[j]) & (df['Label'] == self.label[i])].reset_index(drop=True)

                # Convert 'HeartRate' column to a NumPy array
                arr_list = np.array(df_list[var]['HeartRate'])

                arr_reshape = self.interpolate_and_reshape(arr_list, max_size, frame_rate)


                df_list[var] = pd.DataFrame(arr_reshape)

                # Adding a label to the data segments
                df_list[var]['Label'] = 0 if 'Relax' in self.label[i] else i-3
                df_list[var]['Label_ori'] = self.label[i]
                df_list[var]['Subject'] = self.subjects[j]

                df_final = pd.concat([df_final, df_list[var]])

        return df_final

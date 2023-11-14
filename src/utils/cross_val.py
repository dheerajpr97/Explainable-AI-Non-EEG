import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class TrainTestSplitter:
    def __init__(self, subjects, labels):
        """
        Initializes the TrainTestSplitter class.

        Parameters:
        - subjects (list): List of subjects.
        - labels (list): List of labels.
        """
        self.subjects = subjects
        self.labels = labels

    def train_test_losego(self, df, seg_index=0, num_seg=1):
        """
        Leave one segment out for all modalities.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing raw data.
        - seg_index (int): Index of the segment to be left for the test dataset (default: 0).
        - num_seg (int): Number of segments to include in the test dataset (default: 1).

        Returns:
        - pd.DataFrame, pd.DataFrame: Train and test DataFrames.
        """
        # Initialize empty DataFrames for train and test datasets
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        # Loop through each subject
        for i in range(len(self.subjects)):
            # Loop through each label
            for j in range(len(self.labels)):
                # Get the data for the current subject and label
                data_all = df[(df['Subject'] == self.subjects[i]) & (df['Label_ori'] == self.labels[j])]['Data'].values
                data_all = np.array([np.array(xi) for xi in data_all])

                # Select the test data based on the segment index and number of segments
                data_test = data_all[seg_index:(seg_index) + num_seg] # *8

                # Remove the test data from the training data
                data_train = np.delete(data_all, seg_index, axis=0)

                # Create DataFrames for the train and test data
                df_one_train = self.create_dataframe(data_train)
                df_one_test = self.create_dataframe(data_test)

                # Concatenate the train and test DataFrames
                df_train = pd.concat([df_train, df_one_train]).reset_index(drop=True)
                df_test = pd.concat([df_test, df_one_test]).reset_index(drop=True)

        # Return the train and test DataFrames
        return df_train, df_test

    def train_test_loso(self, df, subject_index=0):
        """
        Leave one subject out for all modalities.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing raw data.
        - subject_index (int): Index of the subject to be selected as the test subject (default: 0).

        Returns:
        - pd.DataFrame, pd.DataFrame: Train and test DataFrames.
        """
        # Initialize empty DataFrames for train and test data
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        # Get train data by excluding the test subject's data from the DataFrame
        data_train = df[(df['Subject'] != self.subjects[subject_index])]['Data'].values 
        data_train = np.array([np.array(xi) for xi in data_train])

        # Get test data for the selected subject
        data_test = df[(df['Subject'] == self.subjects[subject_index])]['Data'].values
        data_test = np.array([np.array(xi) for xi in data_test])

        # Create DataFrames for train and test data
        df_one_train = self.create_dataframe(data_train)
        df_one_test = self.create_dataframe(data_test)

        # Concatenate train and test DataFrames
        df_train = pd.concat([df_train, df_one_train]).reset_index(drop=True)
        df_test = pd.concat([df_test, df_one_test]).reset_index(drop=True)

        return df_train, df_test

    def create_dataframe(self, data):
        """
        Convert NumPy arrays back to a DataFrame.

        Parameters:
        - data (np.ndarray): Input NumPy array.

        Returns:
        - pd.DataFrame: DataFrame created from the NumPy array.
        """
        # Create an empty DataFrame
        df_one = pd.DataFrame()

        # Extract the data from the NumPy array and assign it to the 'Data' column
        df_one['Data'] = list(data[:, :-3, :])

        # Extract the label from the NumPy array and assign it to the 'Label' column
        df_one['Label'] = data[:, -3, 0]

        # Extract the original label from the NumPy array and assign it to the 'Label_ori' column
        df_one['Label_ori'] = data[:, -2, 0]

        # Extract the subject from the NumPy array and assign it to the 'Subject' column
        df_one['Subject'] = data[:, -1, 0]

        # Return the created DataFrame
        return df_one


    def train_val_split(self, df, test_size=0.2):
        """
        Split the DataFrame into train and test sets.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing raw data.
        - test_size (float): Proportion of the dataset to include in the test split (default: 0.2).

        Returns:
        - pd.DataFrame, pd.DataFrame: Train and test DataFrames.
        """
        # Split the DataFrame into train and test sets using the specified test size
        df_train, df_val = train_test_split(df, test_size=test_size)

        # Return the train and test DataFrames
        return df_train, df_val
        

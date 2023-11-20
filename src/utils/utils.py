import pickle

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def load_csv_to_dataframe(file_path, dropna=True):
    """
    Load data from a CSV file and return a cleaned DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        dropna (bool): Whether to drop rows with missing values (default: True).

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    # Load the data from the CSV file
    df = pd.read_csv(file_path)

    # Drop rows with missing values if specified
    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


def dataframe_to_array(df):
    """
    Process the DataFrame and extract arrays of data and labels.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
    - np.ndarray, np.ndarray: Arrays of data and labels.
    """
    data = np.array([np.array(xi, dtype='float32') for xi in df['Data']])
    labels = np.array(df['Label'])

    return data, labels

def save_dataframe_to_pickle(df, file_path):
    """
    Save a DataFrame to a pickle file with correct data types.

    Parameters:
    - df (pd.DataFrame): DataFrame to be saved.
    - file_path (str): File path where the pickle file will be saved.
    """

    # Save the DataFrame to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

def load_dataframe_from_pickle(file_path):
    """
    Load a DataFrame from a pickle file with correct data types.

    Parameters:
    - file_path (str): File path of the pickle file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    # Load the DataFrame from the pickle file
    with open(file_path, 'rb') as file:
        df = pickle.load(file)

    return df

def save_array_to_hdf5(filepath, data, dataset_name='data'):
    """
    Save a NumPy 2D array to an HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file.
        data (numpy.ndarray): The 2D array to be saved.
        dataset_name (str, optional): The name of the dataset in the HDF5 file (default: 'data').
    """
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset(dataset_name, data=data)

def load_array_from_hdf5(filepath, dataset_name='data'):
    """
    Load a NumPy 2D array from an HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file.
        dataset_name (str, optional): The name of the dataset in the HDF5 file (default: 'data').

    Returns:
        numpy.ndarray: The loaded 2D array.
    """
    with h5py.File(filepath, 'r') as hf:
        data = hf[dataset_name][:]
    return data



def create_prediction_dataframe(test_data, predicted_label, class_labels):
    """
    Create a DataFrame to store predictions.

    Args:
    - test_data (numpy.ndarray): The input test data.
    - predicted_label (int): The predicted label.
    - class_labels (list): List of class labels.

    Returns:
    - pandas.DataFrame: The prediction DataFrame.
    """
    pred_dict = {
        'Data': [test_data.squeeze()],
        'Label': predicted_label,
        'Label_ori': class_labels[predicted_label[0]]
    }
    pred_df = pd.DataFrame.from_dict(pred_dict)
    return pred_df

def preprocess_train_val(data, labels, test_size=0.3):
    """
    Preprocess data and labels, and split into train and validation sets.

    Parameters:
    - data (np.ndarray): Array of data.
    - labels (np.ndarray): Array of labels.
    - test_size (float): Proportion of the dataset to include in the test split (default: 0.3).

    Returns:
    - np.ndarray, np.ndarray, np.ndarray, np.ndarray: Processed train and validation data and labels.
    """
    # Split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=test_size, random_state=1, shuffle=True)

    # Normalize labels
    nb_classes = len(np.unique(y_val))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_val = (y_val - y_val.min()) / (y_val.max() - y_val.min()) * (nb_classes - 1)

    # One-hot encode labels
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_val = keras.utils.to_categorical(y_val, nb_classes)

    # Normalize data separately for training and validation sets
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std

    x_val_mean = x_val.mean()
    x_val_std = x_val.std()
    x_val = (x_val - x_val_mean) / x_val_std

    # Reshape data
    x_train = x_train.reshape(x_train.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))

    return x_train, Y_train, x_val, Y_val

def preprocess_test_data(test_data, train_data_mean, train_data_std):
    """
    Normalize the test data using the mean and standard deviation of the training data.

    Args:
    - test_data (numpy.ndarray): The test dataset to be normalized.
    - train_data_mean (float): The mean value of the training data.
    - train_data_std (float): The standard deviation of the training data.

    Returns:
    - numpy.ndarray: The normalized test data.
    """
    # Normalize the test data
    test_data = (test_data - train_data_mean) / train_data_std
 
    # Reshape the test data to include an additional dimension
    test_data = test_data.reshape(test_data.shape + (1,))
    
    return test_data

def evaluate_test_data(test_data, test_labels, model, train_data_mean, train_data_std):
    """
    Evaluate the test data using the trained model.
    
    Args:
        test_data (numpy.ndarray): The test data.
        test_labels (numpy.ndarray): The labels for the test data.
        model (object): The trained model.
    """
    # Preprocess the test data
    preprocessed_test_data = preprocess_test_data(test_data, train_data_mean, train_data_std)
    
    # Predict the labels for the test data
    predictions = model.predict(preprocessed_test_data)
    
    # Convert the predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Define class labels for the confusion matrix
    class_labels = {0: 'Relax', 1: 'PhysicalStress', 2: 'CognitiveStress', 3: 'EmotionalStress'}
    
    # Compute the confusion matrix
    confusion_mat = confusion_matrix(np.int16(test_labels), predicted_labels, normalize='true')
    
    # Compute the accuracy
    accuracy = accuracy_score(np.int16(test_labels), predicted_labels) * 100
    
    # Print the accuracy
    print('Accuracy:', accuracy)
    
    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat / np.sum(confusion_mat, axis=1), annot=True, fmt='.2%', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    
    plt.yticks(ticks=[0, 1, 2, 3], labels=[class_labels[i] for i in range(4)], rotation=0, va='center')
    plt.xticks(ticks=[0, 1, 2, 3], labels=[class_labels[i] for i in range(4)], ha='center')
    
    plt.show()
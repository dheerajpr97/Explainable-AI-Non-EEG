import pandas as pd
import numpy as np
import pickle
import keras
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd

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
    #data_mean = np.mean(data, axis=1)
    labels = np.array(df['Label'])

    return data, labels

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

def save_dataframe_to_pickle(df, file_path):
    """
    Save a DataFrame to a pickle file with correct data types.

    Parameters:
    - df (pd.DataFrame): DataFrame to be saved.
    - file_path (str): File path where the pickle file will be saved.
    """
    # Convert the 'Data' column to a list of lists
    #df['Data'] = df['Data'].apply(lambda x: eval(x))

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

def preprocess_test_data(test_data):
    """
    Preprocesses the given test data by subtracting the mean and dividing by the standard deviation.

    Parameters:
        test_data (ndarray): The test data to be preprocessed.

    Returns:
        ndarray: The preprocessed test data.
    """
    # Normalize the test data
    test_data = (test_data - test_data.mean()) / test_data.std()
 
    # Reshape the test data to include an additional dimension
    test_data = test_data.reshape(test_data.shape + (1,))
    
    return test_data

def evaluate_test_data(test_data, test_labels, model):
    """
    Evaluate the test data using the trained model.
    
    Args:
        test_data (numpy.ndarray): The test data.
        test_labels (numpy.ndarray): The labels for the test data.
        model (object): The trained model.
    """
    # Preprocess the test data
    test_data = preprocess_test_data(test_data)
    
    # Predict the labels for the test data
    pred = model.predict(test_data)
    
    # Convert the predicted probabilities to class labels
    y_pred = np.argmax(pred, axis=1)
    
    # Compute the confusion matrix
    cm = confusion_matrix(np.int16(test_labels), y_pred, normalize='true')
    
    # Compute the accuracy
    acc = accuracy_score(np.int16(test_labels), y_pred) * 100
    
    # Print the accuracy
    print('Accuracy:', acc)
    
    # Plot the normalized confusion matrix
    sns.heatmap(cm/np.sum(cm, axis=1), annot=True, fmt='.2%', cmap='Blues')

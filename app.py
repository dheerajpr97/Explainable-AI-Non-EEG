import datetime
import os

import h5py
import numpy as np
import streamlit as st
import tensorflow as tf

from src.constants import (ACTIVATION_LAYER_NAME, LABELS, LABELS_DICT, MODALS,
                           SAVED_MODEL_DIR, TRAIN_DATA_MEAN, TRAIN_DATA_STD)
from src.utils.grad_cam_pred import GradCAMVisualizer
from src.utils.utils import create_prediction_dataframe, load_array_from_hdf5

# Set the device to CPU to predict on local machine
tf.config.set_visible_devices([], 'GPU')

# Load your trained model, modals, and labels
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'model.h5')
modals = MODALS
labels = LABELS
class_labels = LABELS_DICT

# Streamlit app
st.header("Grad-CAM Visualization with Prediction")

# File upload widget
uploaded_file = st.file_uploader("Upload an HDF5 file", type=["h5"])

if uploaded_file is not None:

    # Load the uploaded file into a NumPy array
    test_data = load_array_from_hdf5(uploaded_file)
    norm_test_data = (test_data - TRAIN_DATA_MEAN) / TRAIN_DATA_STD
    norm_test_data = np.expand_dims(norm_test_data, axis=(0, -1))

    # Display the loaded data
    st.subheader("Data Loaded...")

    # Create buttons for "Predict" and "Display"
    predict_button = st.button("Predict and Display")

    if predict_button:
        # Perform Grad-CAM visualization with prediction
        model = tf.keras.models.load_model(MODEL_PATH)

        start = datetime.datetime.now()
        prediction = model.predict(norm_test_data)
        end = datetime.datetime.now()
        time_diff = end - start
        st.write("Time taken to predict: ", time_diff.total_seconds(), " sec")

        predicted_label_index = np.argmax(prediction, axis=1)
        predicted_label = class_labels[predicted_label_index[0]]
        predicted_prob = prediction[0][predicted_label_index[0]]

        # Display predicted class
        st.subheader("Predicted Class")
        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Predicted Probability: {predicted_prob:.3f}")

        # Create a prediction DataFrame using the function
        prediction_df = create_prediction_dataframe(test_data, predicted_label_index, class_labels)

        # Create a GradCAMVisualizer instance
        visualizer = GradCAMVisualizer(model, ACTIVATION_LAYER_NAME, MODALS, LABELS)

        # Perform Grad-CAM visualization for all modalities
        st.subheader("Grad-CAM Visualization for All Modalities")
        disp = visualizer.vis_grad_cam_all_modal(prediction_df)
        st.image(disp)

        # Perform Grad-CAM visualization for one modality
        st.subheader("Grad-CAM Visualization for One Modality")
        disp = visualizer.vis_grad_cam_one_modal(prediction_df)
        st.image(disp)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.constants import (LABELS, MODALS, SUBJECTS, TRAIN_DATA_MEAN,
                           TRAIN_DATA_STD)


class GradCAMVisualizer:
    def __init__(self, model, model_layer, modals, labels):
        """
        Initialize the GradCAMVisualizer class.
        
        Args:
        - model: The model to be visualized.
        - model_layer: The layer of the model to be visualized.
        - modals: The modals used in the visualization.
        - labels: The labels associated with the modals.
        """
        self.model = model
        self.model_layer = model_layer
        self.modals = modals
        self.labels = labels

    @staticmethod
    def preprocess_data(test_df):
        """
        Preprocess the test data.
        
        Args:
        - test_df: The test dataframe.
        
        Returns:
        - Preprocessed test data.
        """
        x_test = np.array([np.array(xi, dtype='float32') for xi in test_df['Data'].values[0]])
        y_test = test_df['Label']
        y_test_ori = test_df['Label_ori']
        
        x_test = (x_test - TRAIN_DATA_MEAN)/(TRAIN_DATA_STD)
        x_test = x_test.reshape(x_test.shape + (1,))
        
        return x_test, y_test, y_test_ori

    def vis_grad_cam_one_modal(self, test_sample, label_index=0): 
        """
        Plot Grad-CAM visualization for a single test sample.

        Args:
        - test_sample: The test data.
        - label_index: The index of the label to visualize (default: 0).
        """       
        # Get the output of the last convolutional layer
        get_last_conv = keras.backend.function([self.model.layers[0].input], [self.model.layers[-3].output])
        # Get the softmax output
        get_softmax = keras.backend.function([self.model.layers[0].input], [self.model.layers[-1].output])
        # Get the weight of the softmax layer
        softmax_weight = self.model.get_weights()[-2]

        # Iterate over each test sample
        for row in range(len(test_sample)):
            # Preprocess the test data
            x_test, y_test, y_test_ori = self.preprocess_data(test_sample[row:row+1])
            x_test = np.expand_dims(x_test, axis=0)
            # Get the output of the last convolutional layer for the test sample
            last_conv = get_last_conv([x_test])[0]
            # Get the softmax output for the test sample
            softmax = get_softmax([x_test])[0]         
            # Compute the class activation map (CAM)
            CAM = np.dot(last_conv, softmax_weight)
            x_test = (x_test * TRAIN_DATA_STD) + TRAIN_DATA_MEAN

            num_rows = 5
            num_cols = 2
            plot_counter = 1
            plt.figure(figsize=(12, 15))

            # Iterate over each modality
            for modal in range(7):
                # Normalize the CAM
                normalized_CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))

                plt.subplot(num_rows, num_cols, plot_counter)
                plt.title(f"Modality: {self.modals[modal]}  Likelihood of label: {y_test_ori[label_index]}: {softmax[0][int(y_test[label_index])]:.2f}")
                plt.plot(x_test[0, :, modal].squeeze())
                plt.scatter(np.arange(len(x_test[0])), x_test[0, :, modal], 
                            cmap='hot_r', c=normalized_CAM[0, :, modal, int(y_test[label_index])], s=100)
                plt.colorbar()
                plot_counter += 1

            plt.tight_layout()
            plt.show()
            

    def vis_grad_cam_all_modal(self, test_sample, color=False, plot=True):
            """
            Generates a Grad-CAM visualization for all modalities of a given test sample.
            
            Args:
                test_sample (numpy.ndarray): The test sample to generate the visualization for.
                subject (str): The subject of the test sample.
                class_name (str, optional): The class name to use for the visualization. Defaults to 'Relax_1'.
                color (bool, optional): Whether to convert the image to grayscale before generating the visualization. 
                                    Defaults to False.
                plot (bool, optional): Whether to plot the generated heatmap. Defaults to True.
            
            Returns:
                numpy.ndarray: The normalized heatmap of the visualization.
            """
            
            # Preprocess the test data
            x_test, y_test, y_test_ori = self.preprocess_data(test_sample)

            # Convert image to grayscale if needed
            img_gray = cv2.cvtColor(x_test, cv2.COLOR_BGR2GRAY) if color else x_test

            # Expand dimensions of the grayscale image
            img_gray = np.expand_dims(img_gray, axis=0)

            # Get the last convolutional layer and create a heatmap model
            last_conv_layer = self.model.get_layer(self.model_layer)
            heatmap_model = tf.keras.models.Model(inputs=self.model.input,
                                                outputs=[last_conv_layer.output, self.model.output])

            # Compute the gradients and pooled gradients
            with tf.GradientTape() as tape:
                conv_output, predictions = heatmap_model(img_gray)
                loss = predictions[:, np.argmax(predictions[0])]
                grads = tape.gradient(loss, conv_output)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Compute the heatmap
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

            # Convert the input image to RGB and normalize its values
            img_rgb = cv2.cvtColor(x_test, cv2.COLOR_GRAY2RGB)
            img_rgb = (((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())) * 255).astype(np.uint8)

            squeezed_hm = np.squeeze(heatmap)
            normalized_hm = cv2.resize(squeezed_hm, (img_rgb.shape[1], img_rgb.shape[0]))

            # Apply color mapping to the normalized heatmap
            normalized_hm_vis = (255 * normalized_hm).astype(np.uint8)
            normalized_hm_vis = cv2.applyColorMap(normalized_hm_vis, cv2.COLORMAP_JET)

            # Plot the heatmap
            if plot:
                rows, cols = 1, 1
                fig, axs = plt.subplots(rows, cols, figsize=(15, 5))

                # Set yticks and their labels       
                plt.setp(axs, yticks=[0, 1, 2, 3, 4, 5, 6], yticklabels=MODALS)  

                axs1 = axs.imshow(cv2.cvtColor(normalized_hm_vis.transpose(1, 0, 2), cv2.COLOR_BGR2RGB), cmap='jet', 
                                vmin=0, vmax=1)
                axs.set_title(f'Normalized Grad-CAM | {y_test_ori.values[0]}')
                fig.colorbar(axs1, shrink=0.38)

                axs.set_aspect('equal')

            return normalized_hm


    def average_grad_cam(self, df):
        """
        Calculate the average Grad-CAM for each label and plot the results.

        Args:
            df (pandas.DataFrame): Input data.
        """
        # Loop through each label
        for label in range(len(LABELS)):
            norm_hm = []
            
            # Loop through segments per subject
            for i in range(label, 140, 7):
                # Calculate Grad-CAM for each segment
                norm_hm.append(self.vis_grad_cam_all_modal(df[i:i+1], color=False, plot=False))
            
            # Calculate the average Grad-CAM
            label_avg = np.mean(norm_hm, axis=0)
            
            # Plotting the average Grad-CAM
            normalized_hm_vis = (255 * label_avg).astype(np.uint8)
            normalized_hm_vis = cv2.applyColorMap(normalized_hm_vis, cv2.COLORMAP_JET)

            plt.figure(figsize=(15, 5))
            plt.yticks([0, 1, 2, 3, 4, 5, 6], labels=self.modals)
            plt.title('Average Grad-CAM | ' + str(LABELS[label]))
            plt.imshow(cv2.cvtColor(normalized_hm_vis.transpose(1, 0, 2), cv2.COLOR_BGR2RGB), cmap='jet',
                        vmin=0, vmax=1)
            plt.colorbar(shrink=0.38)
            plt.show()

            
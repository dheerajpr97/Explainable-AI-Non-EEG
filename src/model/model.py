import tensorflow as tf
from tensorflow import keras
import os
from src.constants.constants import ROOT_DIR

class ConvModel:
    def __init__(self, input_shape, nb_classes):
        """
        Initializes the ConvModel class.

        Parameters:
        - input_shape (tuple): Shape of the input data (excluding batch dimension).
        - nb_classes (int): Number of classes for the output layer.
        """
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the ConvModel architecture.

        Returns:
        - tf.keras.Model: The ConvModel architecture.
        """
        x_in = keras.layers.Input(self.input_shape)
        
        x = self.conv_block(x_in, 32, 8)
        x = self.conv_block(x, 32, 8)
        
        x = self.conv_block(x, 64, 5)
        x = self.conv_block(x, 64, 5)
        
        x = self.conv_block(x, 256, 3)
        x = self.conv_block(x, 256, 3)
        
        x = keras.layers.Dropout(0.5)(x)
        
        x = self.conv_block(x, 128, 3)
        x = self.conv_block(x, 128, 3)
        
        full = keras.layers.GlobalAveragePooling2D()(x)
        out = keras.layers.Dense(self.nb_classes, activation='softmax')(full)
        
        model = keras.models.Model(inputs=x_in, outputs=out)
        return model

    def conv_block(self, x, filters, kernel_size):
        """
        Constructs a convolutional block.

        Parameters:
        - x (tf.Tensor): Input tensor.
        - filters (int): Number of filters for the convolutional layers.
        - kernel_size (int): Size of the convolutional kernel.

        Returns:
        - tf.Tensor: Output tensor after applying the convolutional block.
        """
        x = keras.layers.Conv2D(filters, kernel_size, 1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x


def train_cp(model, x_train, y_train, x_val, y_val, epochs, batch_size, model_path):
    """
    Trains the ConvModel model.

    Parameters:
    - model (keras.Model): ConvModel model.
    - x_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    - x_val (np.ndarray): Validation data.
    - y_val (np.ndarray): Validation labels.
    - epochs (int): Number of epochs.
    - batch_size (int): Batch size.
    - path (str): Path to save the model checkpoints.

    Returns:
    - keras.callbacks.History: History of the training process.
    """
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=0.000001)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                                                monitor='val_accuracy', save_weights_only=True, 
                                                                mode='max', save_best_only=True)
    
    print('Training...')

    history = model.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, 
                              callbacks=[lr_scheduler, early_stopping, model_checkpoint_callback])

    # Save the model checkpoints
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model.model.save(model_path)
    print('Model saved')
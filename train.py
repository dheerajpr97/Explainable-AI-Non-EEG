from src.dataset.create_dataset import DataframePrepAllMod  
from src.constants.constants import *
from src.model.model import ConvModel, train_cp
from src.utils.cross_val import TrainTestSplitter
from src.utils.utils import *
import tensorflow as tf

import argparse
import os

def main(args):
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    num_epochs = args.e
    batch_size = args.batch
    max_size = args.max_size*args.samp_freq
    
    # Check if dataset file exists and create it if not
    if not os.path.isfile(os.path.join(SAVED_DATASET_DIR, args.f)): 
        print("Dataset file not found!")
        print("Generating dataset file:", args.f)
        create_data_set = DataframePrepAllMod(
            subjects=SUBJECTS, modals=MODALS, modals_1=MODALS_1, modals_2=MODALS_2,
            hr_spo2_file=HR_SPO2_FILE_PATH, acc_temp_eda_file=ACC_TEMP_EDA_FILE_PATH,
            label=LABELS, frame_rate=args.frame_rate, max_size=max_size
        )
        data_set = create_data_set.dataframe_prep_allmod()
        save_dataframe_to_pickle(data_set, os.path.join(SAVED_DATASET_DIR, args.f))

    else:
        print("Dataset file found!")
    
    # Load dataset  
    data_set = load_dataframe_from_pickle(os.path.join(SAVED_DATASET_DIR, args.f))
    cross_val = TrainTestSplitter(subjects=SUBJECTS, labels=LABELS)  

    # Split dataset into train and test depending on cross validation method           
    if args.cross_val == 'LOSO': # Leave one subject out            
        train_df, test_df = cross_val.train_test_loso(data_set, subject_index=args.sub_index)
        train_filename = f"train_data_{args.cross_val}_{args.sub_index}_{args.samp_freq}Hz.pkl"
        test_filename = f"test_data_{args.cross_val}_{args.sub_index}_{args.samp_freq}Hz.pkl"

    else:
        train_df, test_df = cross_val.train_test_losego(data_set, seg_index=args.seg_index, num_seg=args.num_seg, samp_freq=args.samp_freq) # Leave one segment out
        train_filename = f"train_data_{args.cross_val}_{args.seg_index}_{args.num_seg}_{args.samp_freq}Hz.pkl"
        test_filename = f"test_data_{args.cross_val}_{args.seg_index}_{args.num_seg}_{args.samp_freq}Hz.pkl"

    # Save train and test dataframes     
    train_path = os.path.join(SAVED_DATASET_DIR, train_filename)            
    test_path = os.path.join(SAVED_DATASET_DIR, test_filename)
    save_dataframe_to_pickle(train_df, train_path) 
    save_dataframe_to_pickle(test_df, test_path) 

    # Prepare data for training and validation 
    train_data, train_label = dataframe_to_array(train_df)
    train_data, train_label, val_data, val_label = preprocess_train_val(train_data, train_label)
    print("train set size:", len(train_data))
    print("val set size:", len(val_data))
    

    # Model training
    input_shape = train_data.shape[1:]

    if args.cross_val == 'LOSO':
        model_path = os.path.join(SAVED_MODEL_DIR, f"model_{args.cross_val}_{args.sub_index}_{args.samp_freq}Hz.h5")
    else:
        model_path = os.path.join(SAVED_MODEL_DIR, f"model_{args.cross_val}_{args.seg_index}_{args.num_seg}_{args.samp_freq}Hz.h5")

    model = ConvModel(input_shape=input_shape, nb_classes=len(LABELS_DICT.values()))
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.8)
    model.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    train_cp(model, train_data, train_label, val_data, val_label, epochs=num_epochs, batch_size=batch_size, model_path=model_path)
    print("Training Finished")


if __name__ == "__main__":
    
    #for i in os.listdir('data/raw'):
        #if i == "fully_connected_patches_scale_diff_all_shapes__15_kps.pkl":
    frame_rate = 60
    max_size = 360
    dataset_to_train_1Hz = "processed_dataframe_allmod.pkl"
    dataset_to_train_8Hz = "processed_dataframe_allmod_8Hz.pkl"
    
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--f', type=str, default="processed_dataframe_allmod.pkl", help="train dataset file name")
    parser.add_argument('--samp_freq', type=int, default=1, help="sampling frequency in Hz")
    parser.add_argument('--frame_rate', type=int, default=frame_rate, help="frame size for each segment")
    parser.add_argument('--max_size', type=int, default=max_size, help="maximum size for interpolation")
    parser.add_argument('--m', type=str, default=f'ConvModel-{dataset_to_train_1Hz}', help="model name")
    parser.add_argument('--device', type=str, default='cuda', help="cuda / cpu")
    parser.add_argument('--batch', type=int, default=64, help="batch size")
    parser.add_argument('--e', type=int, default=250, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")#1e-3, 5.e-3
    parser.add_argument('--optimizer', type=str, default='SGD', help="optimizer")#1e-3, 5.e-3
    parser.add_argument('--model_dir', type=str, default="models/", help="path to save model")
    parser.add_argument('--cross_val', type=str, default='LOSO', help="cross validation method (LOSO / LOSegO)")
    parser.add_argument('--sub_index', type=int, default=0, help="subject index for LOSO")
    parser.add_argument('--num_seg', type=int, default=1, help="number of segments to leave for LOSegO")
    parser.add_argument('--seg_index', type=int, default=0, help="segment index for LOSegO")
    args = parser.parse_args()
    main(args)
    print("Training Finished")
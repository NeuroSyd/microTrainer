from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model, get_model_student
import argparse
from datasets_shuffle import ECGSequence

import os

import keras
import tensorflow as tf
from keras import layers
# from keras import ops
import numpy as np


def custom_loss(y_true, y_pred):
    # Modify y_true and y_pred to include only the first 3 terms and the 5th term
    y_true_modified = tf.concat([y_true[:, :3], y_true[:, 4:5]], axis=1)
    y_pred_modified = tf.concat([y_pred[:, :3], y_pred[:, 4:5]], axis=1)

    # Calculate binary cross-entropy loss using the modified y_true and y_pred
    loss = tf.keras.losses.binary_crossentropy(y_true_modified, y_pred_modified, from_logits=True)

    return loss

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('--path_to_hdf5', type=str, default='...',
                        help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_csv', type=str, default='...',
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.2')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--file_name', type=str, default='testing',
                        help='file for saving the model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Experiment Number.')
    parser.add_argument('--exp', type=str, default='test',
                        help='Experiment Number.')

    args = parser.parse_args()
    # Optimization settings
    lr = 0.001
    batch_size = args.batch_size
    opt = Adam(lr)
    epochs = 30
    file_name = args.file_name
    save_dir = './tests/'+ file_name

    def check_directory_existence(directory):
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            exit()  # Stop the program if the directory doesn't exist
        else:
            print(f"Directory '{directory}' exists.")

    check_directory_existence(save_dir)

    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)

    callbacks = [
        ReduceLROnPlateau(monitor='val_auc_1', factor=0.1, patience = 3, mode='auto', min_lr=lr / 100),
        EarlyStopping(monitor='val_auc_1',
                      patience = 6,  # Patience should be larger than the one in ReduceLROnPlateau
                      min_delta=0.00001,
                      mode='auto',
                      restore_best_weights = True,)
    ]

    # Create log
    callbacks += [CSVLogger(f"{save_dir}/training_finetune_{args.exp}.csv", append=False)]  # Change append to true if continuing training


    model = tf.keras.saving.load_model(save_dir + '/model.hdf5', compile=False)


    # Set all layers trainable
    for layer in model.layers:
        layer.trainable = True

    # Freeze all layers except the first 3 and the last layer
    for layer in model.layers[3:-1]:
        layer.trainable = False

    model.compile(optimizer=opt,
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Recall(), keras.metrics.Precision(), keras.metrics.AUC(curve='ROC'), keras.metrics.AUC(curve='PR'), keras.metrics.FalseNegatives()],
        loss = custom_loss
    )
    model.summary()

    history = model.fit(train_seq,
                        epochs=epochs,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)

    model.save(f"{save_dir}/model_finetune_{args.exp}.hdf5")
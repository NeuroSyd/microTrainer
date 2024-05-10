from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model, get_model_student
import argparse
from datasets import ECGSequence

import os

import keras
import tensorflow as tf
from keras import layers
# from keras import ops
import numpy as np
import csv


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('--path_to_hdf5', type=str, default='...',
                        help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_csv', type=str, default='...',
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--file_name', type=str, default='testing',
                        help='file for saving the model')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha for KD loss')
    args = parser.parse_args()
    # Optimization settings
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    epochs = 100
    file_name = args.file_name
    save_dir = './tests/'+ file_name
    alpha = args.alpha


    def create_directory_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")


    create_directory_if_not_exists(save_dir)

    # Record alpha in a CSV file in save_dir
    csv_file = os.path.join(save_dir, 'alpha.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['alpha'])
        writer.writerow([alpha])


    callbacks = [
        ReduceLROnPlateau(monitor='val_auc_1', factor=0.1, patience = 10, mode='max', min_lr=lr / 100),
        EarlyStopping(monitor='val_auc_1',
                      patience = 15,  # Patience should be larger than the one in ReduceLROnPlateau
                      min_delta=0.00001,
                      mode='max',
                      restore_best_weights = True,)
    ]

    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)


    # Create log
    callbacks += [ # TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger(save_dir + '/training.csv', append=False)]  # Change append to true if continuing training


    class Distiller(keras.Model):
        def __init__(self, student, teacher):
            super().__init__()
            self.teacher = teacher
            self.student = student

        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            # distillation_bce_fn,
            alpha=0.5,
            # beta = 0.4,
            temperature=3,
        ):
            """Configure the distiller.

            Args:
                optimizer: Keras optimizer for the student weights
                metrics: Keras metrics for evaluation
                student_loss_fn: Loss function of difference between student
                    predictions and ground-truth
                distillation_loss_fn: Loss function of difference between soft
                    student predictions and soft teacher predictions
                alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
                temperature: Temperature for softening probability distributions.
                    Larger temperature gives softer distributions.
            """
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            # self.distillation_bce_fn = distillation_bce_fn
            self.alpha = alpha
            # self.beta = beta
            self.temperature = temperature

        def compute_loss(
            self, x = None, y = None, y_pred = None, sample_weight=None, allow_empty=False
        ):
            teacher_pred = self.teacher(x, training=False)
            student_loss = self.student_loss_fn(y, y_pred)



            teacher_probs = tf.nn.sigmoid(teacher_pred)
            threshold = 0.5
            teacher_probs = tf.cast(teacher_probs > threshold, tf.float32)


            distillation_loss = self.distillation_loss_fn(teacher_probs, y_pred)



            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

            # loss = student_loss
            # print(loss)


            return loss

        def call(self, x):
            return self.student(x)



    # Create the teacher
    teacher = get_model(train_seq.n_classes)
    # Load pre-trained Teacher Model
    teacher_model= tf.keras.saving.load_model('/mnt/data13_16T/jim/ECG/Codes/Transfer_Learning/model/model.hdf5', compile=False)

    teacher.set_weights(teacher_model.get_weights())


    # Create the student
    student = get_model_student(train_seq.n_classes)

    # student.summary()

    # Weight Casting
    # Function to print the number of weights in a layer
    def print_weights_info(layer_name, weights):
        num_weights = len(weights[0].flatten())
        print(f"Number of weights in layer '{layer_name}': {num_weights}")


    # Initialize an empty list to store the weights of the first few layers
    source_first_layers_weights = []

    # Loop through the layers from index 0 to index 2
    for i in range(0, 3):
        # Get the weights of the current layer
        weights = teacher.layers[i].get_weights()

        # Check if the weights are not empty before appending them
        if weights:
            # Set the weights of the current layer in the student model
            student.layers[i].set_weights(weights)

            # Print the number of weights casted for the current layer
            print_weights_info(teacher.layers[i].name, weights)

            # Append the weights to the list
            source_first_layers_weights.append(weights)
        else:
            print(f"Warning: No weights found in layer {i} of the teacher model.")

    # Set the weights of the last layer in the student model
    last_layer_weights = teacher.layers[-1].get_weights()
    student.layers[-1].set_weights(last_layer_weights)

    # Print the number of weights casted for the last layer
    print_weights_info(teacher.layers[-1].name, last_layer_weights)

    # Calculate the total number of weights transferred
    num_weights_transferred = sum(len(weights[0].flatten()) for weights in source_first_layers_weights) + len(
        last_layer_weights[0].flatten())

    # Print the total number of weights transferred
    print("Total number of weights transferred:", num_weights_transferred)

    # Freeze the first 3 layers and the last layer
    for layer in student.layers[:3]:
        layer.trainable = False
    for layer in student.layers[-1:]:
        layer.trainable = False


    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=opt,
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Recall(), keras.metrics.Precision(), keras.metrics.AUC(curve='ROC'), keras.metrics.AUC(curve='PR'), keras.metrics.FalseNegatives()],
        student_loss_fn= keras.losses.BinaryCrossentropy(from_logits=True), # label_smoothing=0.2
        # distillation_loss_fn=keras.losses.KLDivergence(),
        distillation_loss_fn= keras.losses.BinaryCrossentropy(from_logits=True),
        alpha=alpha,
        # beta = 0.4,
        temperature=3,
    )

    # Distill teacher to student

    history = distiller.fit(train_seq,
                      epochs=epochs,
                      initial_epoch=0,  # If you are continuing a interrupted section change here
                      callbacks=callbacks,
                      validation_data=valid_seq,
                      verbose=1
                            )


    student.save(save_dir + "/model.hdf5")
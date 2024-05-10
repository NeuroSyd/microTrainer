import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence
import pandas as pd
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--path_to_hdf5', type=str, default='...',
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model',
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    # parser.add_argument('--output_file', default="./model/ouput.csv",  # or predictions_date_order.csv
    #                     help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--exp', type=str, default='1',
                        help='Experiment Number.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    path = args.path_to_model
    # path_to_model = path + '/model_finetune.hdf5'
    path_to_model = f"{path}/model_finetune_{args.exp}.hdf5"
    # path_to_model = path + '/model.hdf5'


    path_to_csv = '...'

    # Read the CSV file
    label = pd.read_csv(path_to_csv)[['1dAVb', 'RBBB', 'LBBB', 'AF']]
    # Get the column names
    columns = label.columns
    # Convert label values to np.float32 data type
    y = label.values.astype(np.float32)[6000: , :]

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model('...', compile=False)
    model_cast = load_model(path_to_model, compile=False)


    def print_weights_info(layer_name, weights):
        num_weights = len(weights[0].flatten())
        print(f"Number of weights in layer '{layer_name}': {num_weights}")


    source_first_layers_weights = []

    for i in range(0, 3):
        # Get the weights of the current layer
        weights = model_cast.layers[i].get_weights()

        # Check if the weights are not empty before appending them
        if weights:
            # Set the weights of the current layer in the student model
            model.layers[i].set_weights(weights)

            # Print the number of weights casted for the current layer
            print_weights_info(model_cast.layers[i].name, weights)

            # Append the weights to the list
            source_first_layers_weights.append(weights)
        else:
            print(f"Warning: No weights found in layer {i} of the teacher model.")

    # Set the weights of the last layer in the student model
    last_layer_weights = model_cast.layers[-1].get_weights()
    model.layers[-1].set_weights(last_layer_weights)

    # Print the number of weights casted for the last layer
    print_weights_info(model_cast.layers[-1].name, last_layer_weights)

    # Calculate the total number of weights transferred
    num_weights_transferred = sum(len(weights[0].flatten()) for weights in source_first_layers_weights) + len(
        last_layer_weights[0].flatten())

    # Print the total number of weights transferred
    print("Total number of weights transferred:", num_weights_transferred)






    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)[6000: , :]

    # Generate dataframe
    df = pd.DataFrame(y_score)
    df.to_csv(f"{path}/y_pred_{args.exp}.csv", index=False,
              header=False)  # Set index=False and header=False to exclude row and column headers

    print("Output predictions saved")


    y_pred = y_score[:, [0, 1, 2, 4]]
    y_true = y

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
    import csv

    # set a threshold value of 0.5
    threshold = 0.5

    # apply the threshold to convert the predicted probabilities to binary values
    y_pred_bin = np.where(y_pred >= threshold, 1, 0)

    # calculate evaluation metrics
    precision = precision_score(y_true, y_pred_bin, average=None)
    recall = recall_score(y_true, y_pred_bin, average=None)
    f1 = f1_score(y_true, y_pred_bin, average=None)
    auroc_scores = roc_auc_score(y_true, y_pred, average=None)

    auprc_scores = []
    for i in range(y_true.shape[1]):
        precision_class, recall_class, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        auprc_class = auc(recall_class, precision_class)
        auprc_scores.append(auprc_class)

    # print evaluation metrics
    print("Class:", columns)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("AUROC:", auroc_scores)
    print("AUPRC:", auprc_scores)

    # Write evaluation metrics to a CSV file
    csv_name = f"{path}/performance_{args.exp}.csv"

    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'AUPRC'])
        for i in range(len(columns)):
            writer.writerow([columns[i], precision[i], recall[i], f1[i], auroc_scores[i], auprc_scores[i]])

        # Write average metrics row
        writer.writerow(['Average', sum(precision) / len(precision), sum(recall) / len(recall), sum(f1) / len(f1),
                         sum(auroc_scores) / len(auroc_scores), sum(auprc_scores) / len(auprc_scores)])

    print('Completed')
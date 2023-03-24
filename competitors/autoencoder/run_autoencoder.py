import os
import numpy as np
import pandas as pd
import sys
import time
from autoencoder import train_single_model, anomaly_detection
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

n_clusters = [5, 10, 25, 50, 100]

columns = ["date","ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
           "DestinationAimedArrivalTime","OriginAimedDepartureTime","HeadwayService_False","anomaly"]

threshold = [1.5, 3.0]

ae_encoding_dim = [2, 4]

ae_epochs = 50
ae_batch_size = 32
ae_l_rate = 0.0001


for n in n_clusters:
    dataset_name = "aggr_" + str(n)
    filename_data_train = "Datasets/" + dataset_name + ".txt.red.txt"
    filename_data_pred = "Datasets/" + dataset_name + ".txt.red_test.txt"

    data_train_full = pd.read_csv(filename_data_train,
                parse_dates=['date'],
                na_values=[0.0])

    data_train_full.fillna(0, inplace=True)

    data_train_full = data_train_full.loc[:, columns]
    #________________________________________________________
    data_pred_full = pd.read_csv(filename_data_pred,
                parse_dates=['date'],
                na_values=[0.0])

    data_pred_full.fillna(0, inplace=True)

    ground_truth_cols = ['anomaly']

    data_gt = data_pred_full.loc[:, ground_truth_cols]

    data_pred_full = data_pred_full.loc[:, columns]
    #________________________________________________________

    for emb in ae_encoding_dim:
        for t in threshold:
            setting_name = dataset_name + "_AE_encoding_dim_" + str(emb) + "_threshold_" + str(t)

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            metrics_ae_weighted = []
            metrics_ae_micro = []
            metrics_ae_macro = []

            list_of_rows_ae = []
            pred_real_ae = []

            train_data = data_train_full.drop(['date'], axis=1)
            print("Selected training data length: " + str(len(train_data)))

            # current day data for prediction
            data_pred = data_pred_full.drop(['date'], axis=1)
            print(np.shape(data_pred))

            # ground truth (anomaly yes/ no)
            real_classes = data_gt.copy()
            real_classes = np.array(real_classes).flatten()


            #*** AE ***

            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler_fit = min_max_scaler.fit(train_data)
            features = len(columns)

            start = time.time()
            ae_model = train_single_model(train_data, emb, ae_epochs, ae_batch_size, ae_l_rate,
                                                 min_max_scaler_fit, features, 0)
            print("AE trained")
            stop = time.time()
            print(f"Training time: {stop - start}s")
            print("Predicting on test with reconstruction mode and T=" + str(t))
            ae_preds, clean_data = anomaly_detection(ae_model, data_pred, t, min_max_scaler_fit)


            confusion_matrix_ae = confusion_matrix(real_classes, ae_preds)
            ae_tn, ae_fp, ae_fn, ae_tp = confusion_matrix_ae.ravel()

            print("Anomaly detection with AE model:")

            [precision_ae_weighted, recall_ae_weighted, fscore_ae_weighted, support_ae_weighted] = precision_recall_fscore_support(real_classes, ae_preds, average='weighted')
            [precision_ae_micro, recall_ae_micro, fscore_ae_micro, support_ae_micro] = precision_recall_fscore_support(real_classes, ae_preds, average='micro')
            [precision_ae_macro, recall_ae_macro, fscore_ae_macro, support_ae_macro] = precision_recall_fscore_support(real_classes, ae_preds, average='macro')

            metrics_ae_weighted.append(str(precision_ae_weighted) + "," + str(recall_ae_weighted) + "," + str(fscore_ae_weighted))
            metrics_ae_micro.append(str(precision_ae_micro) + "," + str(recall_ae_micro) + "," + str(fscore_ae_micro))
            metrics_ae_macro.append(str(precision_ae_macro) + "," + str(recall_ae_macro) + "," + str(fscore_ae_macro))

            row_ae = [ae_tn, ae_fp, ae_fn, ae_tp]

            list_of_rows_ae.append(row_ae)

            # Saving predicted and real classes
            pred_real_ae = np.hstack((np.reshape(ae_preds, (np.shape(ae_preds)[0], 1)),
                                    np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

            np.savetxt('logs/' + setting_name + "/ae.preds" + ".log", pred_real_ae, delimiter=',', fmt='%s')



            # Saving confusion matrices
            import csv

            with open('logs/' + setting_name + '/ae.matrix.csv', "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(list_of_rows_ae)

            # Saving metrics
            np.savetxt('logs/' + setting_name + "/ae.metrics.weighted.log", metrics_ae_weighted, delimiter=',', fmt='%s')
            np.savetxt('logs/' + setting_name + "/ae.metrics.macro.log", metrics_ae_macro, delimiter=',', fmt='%s')
            np.savetxt('logs/' + setting_name + "/ae.metrics.micro.log", metrics_ae_micro, delimiter=',', fmt='%s')


            #sys.exit(1)

import os
import numpy as np
import pandas as pd
import sys
from autoencoder import train_single_model, anomaly_detection
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

dataset_name = "PV_ITALY"
#dataset_name = "WIND_NREL"


filename_data_pred = "data/" + dataset_name + "_HOURLY_25_50.csv"
days_hist = [30, 60, 90]

threshold = [1.5, 3.0]

estimators = [10, 20, 50]

ae_encoding_dim = [4, 8]

ae_epochs = 50
ae_batch_size = 32
ae_l_rate = 0.0001

# data pipeline
filename_dates = "dates/" + dataset_name + "_100.txt"
dates = pd.read_csv(filename_dates, delimiter=' ')

filename_data_train = "data/" + dataset_name + "_HOURLY_NO_ANOMALIES.csv"
filename_data_pred = "data/" + dataset_name + "_HOURLY_25_50.csv"


if (dataset_name == "PV_ITALY"):
    idplant = 'idsito'
    columns = [idplant, 'data', 'lat', 'lon', 'day', 'ora', 'temperatura_ambiente', 'irradiamento', 'altitude', 'azimuth',
           'pressure', 'windspeed', 'humidity', 'icon', 'dewpoint', 'windbearing', 'cloudcover', 'kwh']
else:
    idplant = 'idplant'
    columns = [idplant, 'data', 'lat', 'lon', 'day', 'hour', 'temperature', 'pressure', 'windspeed', 'humidity',
               'dewpoint', 'windbearing', 'cloudcover', 'power']

data_train_full = pd.read_csv(filename_data_train,
            parse_dates=['data'],
            #index_col=['data'],
            na_values=[0.0])

data_train_full.fillna(0, inplace=True)

data_train_full = data_train_full.loc[:, columns]
#________________________________________________________
data_pred_full = pd.read_csv(filename_data_pred,
            parse_dates=['data'],
            na_values=[0.0])

data_pred_full.fillna(0, inplace=True)

ground_truth_cols = [idplant, 'data', 'anomaly']
data_gt = data_pred_full.loc[:, ground_truth_cols]

data_pred_full = data_pred_full.loc[:, columns]
#________________________________________________________

for emb in ae_encoding_dim:
    for t in threshold:
        for days in days_hist:

            setting_name = dataset_name + "_" + str(days) + "_AE_encoding_dim_" + str(emb) + "_threshold_" + str(t)

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            metrics_ae_weighted = []
            metrics_ae_micro = []
            metrics_ae_macro = []

            list_of_rows_ae = []
            pred_real_ae = []

            test_day = 1

            for pred_day in dates:

                print()

                start_date_pred = datetime.strptime(pred_day, '%Y-%m-%d')
                end_date_pred = start_date_pred + timedelta(days=1)

                print("Prediction day: " + str(start_date_pred))

                start_date_train = start_date_pred - timedelta(days=days+1)
                end_date_train = start_date_pred - timedelta(days=1)

                print("Training start day: " + str(start_date_train))
                print("Training end day: " + str(end_date_train))
                print("Full data length: " + str(len(data_train_full)))

                train_data = data_train_full.loc[(data_train_full['data'] >= start_date_train) & (data_train_full['data'] <= end_date_train)]
                train_data = train_data.drop(['data'], axis=1)
                print("Selected training data length: " + str(len(train_data)))

                # current day data for prediction
                data_pred = data_pred_full.loc[(data_pred_full['data'] >= start_date_pred) & (data_pred_full['data'] <= end_date_pred)]
                print(np.shape(data_pred))

                # ground truth (anomaly yes/ no)
                real_classes = data_gt.loc[(data_gt['data'] >= start_date_pred) & (data_gt['data'] <= end_date_pred)]
                real_classes = real_classes.drop(['data',idplant], axis=1)
                real_classes = np.array(real_classes).flatten()

                pred_data_without_date = data_pred.drop(['data'], axis=1)

                #*** AE ***

                min_max_scaler = preprocessing.MinMaxScaler()
                min_max_scaler_fit = min_max_scaler.fit(train_data)
                features = len(columns)

                ae_model = train_single_model(train_data, emb, ae_epochs, ae_batch_size, ae_l_rate,
                                                 min_max_scaler_fit, features, 0)
                print("AE trained")
                print("Predicting on test with reconstruction mode and T=" + str(t))
                ae_preds, clean_data = anomaly_detection(ae_model, pred_data_without_date, t, min_max_scaler_fit)


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

                np.savetxt('logs/' + setting_name + "/ae.preds_" + str(test_day) + ".log", pred_real_ae, delimiter=',', fmt='%s')
                test_day = test_day + 1


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

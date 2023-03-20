import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

#dataset_name = "PV_ITALY"
dataset_name = "WIND_NREL"


filename_data_pred = "data/" + dataset_name + "_HOURLY_25_50.csv"
days_hist = [30, 60, 90]

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

nu = [0.1, 0.5, 1.0]
kernel = ["linear","rbf"]
gamma = 'auto'

for n in nu:
    for k in kernel:
        one_class_svm = OneClassSVM(nu=n, kernel=k, gamma=gamma)

        for days in days_hist:

            setting_name = dataset_name + "_" + str(days) + "_OCSVM_nu_" + str(n) + "_kernel_" + str(k)

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            metrics_ocsvm_weighted = []
            metrics_ocsvm_micro = []
            metrics_ocsvm_macro = []

            list_of_rows_ocsvm = []
            pred_real_ocsvm = []

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

                ocsvm_model = one_class_svm.fit(train_data)
                print("OCSVM trained")

                ocsvm_predictions = ocsvm_model.predict(pred_data_without_date)
                ocsvm_predictions = np.where(ocsvm_predictions == 1, 0, ocsvm_predictions)
                ocsvm_predictions = np.where(ocsvm_predictions == -1, 1, ocsvm_predictions)

                pred_real_lof = np.hstack(
                    (np.reshape(ocsvm_predictions, (np.shape(ocsvm_predictions)[0], 1)),
                     np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

                confusion_matrix_ocsvm = confusion_matrix(real_classes, ocsvm_predictions)
                ocsvm_tn, ocsvm_fp, ocsvm_fn, ocsvm_tp = confusion_matrix_ocsvm.ravel()

                [precision_ocsvm_weighted, recall_ocsvm_weighted, fscore_ocsvm_weighted, support_ocsvm_weighted] = precision_recall_fscore_support(real_classes, ocsvm_predictions, average='weighted')
                [precision_ocsvm_micro, recall_ocsvm_micro, fscore_ocsvm_micro, support_ocsvm_micro] = precision_recall_fscore_support(real_classes, ocsvm_predictions, average='micro')
                [precision_ocsvm_macro, recall_ocsvm_macro, fscore_ocsvm_macro, support_ocsvm_macro] = precision_recall_fscore_support(real_classes, ocsvm_predictions, average='macro')

                print("Anomaly detection with OCSVM model..")
                metrics_ocsvm_weighted.append(str(precision_ocsvm_weighted) + "," + str(recall_ocsvm_weighted) + "," + str(fscore_ocsvm_weighted))
                metrics_ocsvm_micro.append(str(precision_ocsvm_micro) + "," + str(recall_ocsvm_micro) + "," + str(fscore_ocsvm_micro))
                metrics_ocsvm_macro.append(str(precision_ocsvm_macro) + "," + str(recall_ocsvm_macro) + "," + str(fscore_ocsvm_macro))

                row_ocsvm = [ocsvm_tn,ocsvm_fp, ocsvm_fn, ocsvm_tp]

                list_of_rows_ocsvm.append(row_ocsvm)

                # Saving predicted and real classes
                pred_real_ocsvm = np.hstack((np.reshape(ocsvm_predictions, (np.shape(ocsvm_predictions)[0], 1)),
                                          np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

                np.savetxt('logs/' + setting_name + "/ocsvm.preds_" + str(test_day) + ".log", pred_real_ocsvm, delimiter=',', fmt='%s')
                test_day = test_day + 1

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            # Saving confusion matrices
            import csv

            with open('logs/' + setting_name + '/ocsvm.matrix.csv', "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(list_of_rows_ocsvm)

            # Saving predicted and real classes
            np.savetxt('logs/' + setting_name + "/lof.preds.log", pred_real_ocsvm, delimiter=',', fmt='%s')

            # Saving metrics
            np.savetxt('logs/' + setting_name + "/ocsvm.metrics.weighted.log", metrics_ocsvm_weighted, delimiter=',', fmt='%s')
            np.savetxt('logs/' + setting_name + "/ocsvm.metrics.macro.log", metrics_ocsvm_macro, delimiter=',', fmt='%s')
            np.savetxt('logs/' + setting_name + "/ocsvm.metrics.micro.log", metrics_ocsvm_micro, delimiter=',', fmt='%s')


            #sys.exit(1)

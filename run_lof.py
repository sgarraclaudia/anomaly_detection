import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

#dataset_name = "PV_ITALY"
dataset_name = "WIND_NREL"


filename_data_pred = "data/" + dataset_name + "_HOURLY_25_50.csv"
days_hist = [30, 60, 90]
num_estimators = [10, 20, 50]

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

n_neig = [10, 20, 40]
novelty = [True]

for neig in n_neig:
    for nov in novelty:
        local_outlier_factor = LocalOutlierFactor(n_neighbors=neig, novelty=nov)

        for days in days_hist:

            setting_name = dataset_name + "_" + str(days) + "_LOF_num_neig_" + str(neig) + "_novelty_" + str(nov)

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            metrics_lof_weighted = []
            metrics_lof_micro = []
            metrics_lof_macro = []

            list_of_rows_lof = []
            pred_real_lof = []

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

                local_outlier_factor_model = local_outlier_factor.fit(train_data)
                print("Local outlier factor trained")

                lof_predictions = local_outlier_factor_model.predict(pred_data_without_date)
                lof_predictions = np.where(lof_predictions == 1, 0, lof_predictions)
                lof_predictions = np.where(lof_predictions == -1, 1, lof_predictions)

                pred_real_lof = np.hstack(
                    (np.reshape(lof_predictions, (np.shape(lof_predictions)[0], 1)),
                     np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

                confusion_matrix_lof = confusion_matrix(real_classes, lof_predictions)
                lof_tn, lof_fp, lof_fn, lof_tp = confusion_matrix_lof.ravel()

                [precision_lof_weighted, recall_lof_weighted, fscore_lof_weighted, support_lof_weighted] = precision_recall_fscore_support(real_classes, lof_predictions, average='weighted')
                [precision_lof_micro, recall_lof_micro, fscore_lof_micro, support_lof_micro] = precision_recall_fscore_support(real_classes, lof_predictions, average='micro')
                [precision_lof_macro, recall_lof_macro, fscore_lof_macro, support_lof_macro] = precision_recall_fscore_support(real_classes, lof_predictions, average='macro')

                print("Anomaly detection with Local Outlier Factor model..")
                metrics_lof_weighted.append(str(precision_lof_weighted) + "," + str(recall_lof_weighted) + "," + str(fscore_lof_weighted))
                metrics_lof_micro.append(str(precision_lof_micro) + "," + str(recall_lof_micro) + "," + str(fscore_lof_micro))
                metrics_lof_macro.append(str(precision_lof_macro) + "," + str(recall_lof_macro) + "," + str(fscore_lof_macro))

                row_lof = [lof_tn, lof_fp, lof_fn, lof_tp]

                list_of_rows_lof.append(row_lof)

                # Saving predicted and real classes
                pred_real_lof = np.hstack((np.reshape(lof_predictions, (np.shape(lof_predictions)[0], 1)),
                                          np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

                np.savetxt('logs/' + setting_name + "/lof.preds_" + str(test_day) + ".log", pred_real_lof, delimiter=',', fmt='%s')
                test_day = test_day + 1

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            # Saving confusion matrices
            import csv

            with open('logs/' + setting_name + '/lof.matrix.csv', "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(list_of_rows_lof)

            # Saving predicted and real classes
            np.savetxt('logs/' + setting_name + "/lof.preds.log", pred_real_lof, delimiter=',', fmt='%s')

            # Saving metrics
            np.savetxt('logs/' + setting_name + "/lof.metrics.weighted.log", metrics_lof_weighted, delimiter=',', fmt='%s')
            np.savetxt('logs/' + setting_name + "/lof.metrics.macro.log", metrics_lof_macro, delimiter=',', fmt='%s')
            np.savetxt('logs/' + setting_name + "/lof.metrics.micro.log", metrics_lof_micro, delimiter=',', fmt='%s')


            #sys.exit(1)

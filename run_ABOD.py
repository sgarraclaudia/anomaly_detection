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
from pyod.models.abod import ABOD

np.set_printoptions(threshold=sys.maxsize)


dataset_name = "PV_ITALY"
#dataset_name = "WIND_NREL"


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

clf = ABOD()

for days in days_hist:

    setting_name = dataset_name + "_" + str(days) + "_ABOD"

    if not os.path.exists('logs/' + setting_name):
        os.makedirs('logs/' + setting_name)

    metrics_ABOD_weighted = []
    metrics_ABOD_micro = []
    metrics_ABOD_macro = []

    list_of_rows_ABOD = []
    pred_real_ABOD = []

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

        ABOD_model = clf.fit(train_data)
        print("ABOD model trained")

        ABOD_model_predictions = ABOD_model.predict(pred_data_without_date)
        ABOD_model_predictions = np.where(ABOD_model_predictions == 1, 0, ABOD_model_predictions)
        ABOD_model_predictions = np.where(ABOD_model_predictions == -1, 1, ABOD_model_predictions)

        pred_real_ABOD = np.hstack(
            (np.reshape(ABOD_model_predictions, (np.shape(ABOD_model_predictions)[0], 1)),
             np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

        confusion_matrix_ABOD = confusion_matrix(real_classes, ABOD_model_predictions, labels=[0, 1])
        ABOD_tn, ABOD_fp, ABOD_fn, ABOD_tp = confusion_matrix_ABOD.ravel()

        [precision_ABOD_weighted, recall_ABOD_weighted, fscore_ABOD_weighted, support_ABOD_weighted] = precision_recall_fscore_support(real_classes, ABOD_model_predictions, average='weighted')
        [precision_ABOD_micro, recall_if_micro, fscore_if_micro, support_if_micro] = precision_recall_fscore_support(real_classes, ABOD_model_predictions, average='micro')
        [precision_ABOD_macro, recall_if_macro, fscore_if_macro, support_if_macro] = precision_recall_fscore_support(real_classes, ABOD_model_predictions, average='macro')

        print("Anomaly detection with ABOD model..")
        metrics_ABOD_weighted.append(str(precision_ABOD_weighted) + "," + str(recall_ABOD_weighted) + "," + str(fscore_ABOD_weighted))
        metrics_ABOD_micro.append(str(precision_ABOD_micro) + "," + str(recall_if_micro) + "," + str(fscore_if_micro))
        metrics_ABOD_macro.append(str(precision_ABOD_macro) + "," + str(recall_if_macro) + "," + str(fscore_if_macro))

        row_if = [ABOD_tn, ABOD_fp, ABOD_fn, ABOD_tp]

        list_of_rows_ABOD.append(row_if)

        # Saving predicted and real classes
        pred_real_if = np.hstack((np.reshape(ABOD_model_predictions, (np.shape(ABOD_model_predictions)[0], 1)),
                                  np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

        np.savetxt('logs/' + setting_name + "/ABOD.preds_" + str(test_day) + ".log", pred_real_if, delimiter=',', fmt='%s')
        test_day = test_day + 1

    if not os.path.exists('logs/' + setting_name):
        os.makedirs('logs/' + setting_name)

    # Saving confusion matrices
    import csv

    with open('logs/' + setting_name + '/ABOD.matrix.csv', "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_rows_ABOD)

    # Saving predicted and real classes
    np.savetxt('logs/' + setting_name + "/ABOD.preds.log", pred_real_ABOD, delimiter=',', fmt='%s')

    # Saving metrics
    np.savetxt('logs/' + setting_name + "/ABOD.metrics.weighted.log", metrics_ABOD_weighted, delimiter=',', fmt='%s')
    np.savetxt('logs/' + setting_name + "/ABOD.metrics.macro.log", metrics_ABOD_macro, delimiter=',', fmt='%s')
    np.savetxt('logs/' + setting_name + "/ABOD.metrics.micro.log", metrics_ABOD_micro, delimiter=',', fmt='%s')


    #sys.exit(1)

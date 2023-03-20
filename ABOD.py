import os
import numpy as np
import pandas as pd
import sys
from sklearn.svm import OneClassSVM
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from pyod.models.abod import ABOD

np.set_printoptions(threshold=sys.maxsize)

datasets = ["pd_pstrians"]
#datasets = ["oslo", "oslomobilities", "pd_cars","pd_pstrians"]

clf = ABOD()

numFoldsDict = {
    "oslo": 167,
    "oslomobilities": 1,
    "pd_cars": 358,
    "pd_pstrians": 838
}

datasetSchema = {
    "oslo": ["TimeWindow", "ClusterLatitude", "ClusterLongitude", "Numberofvehicles", "AvgDelay", "PercInPanic",
             "PercInCongestion", "Cluster", "AvgMonitoredCall_VisitNumber", "dateTimedayofTheWeek", "dateTimeDAY",
             "dateTimeHour", "OriginAimedDepartureTimeHour", "HeadwayService_False", "anomaly"],
    "oslomobilities": ["interval_start_time", "X", "Y", "dateTimeDayofTheWeek", "year", "month", "day", "hour",
                       "minute", "people", "anomaly"],
    "pd_cars": ["start_time", "X", "Y", "cameraname", "no_approaching", "no_leaving", "no_unknown",
                    "dateTimeDayofTheWeek", "MONTH", "DAY", "HOUR", "MINUTE", "anomaly"],
    "pd_pstrians": ["start_time", "X", "Y", "sensor_id", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "week_day",
                    "count_people_in", "count_people_out", "holiday", "anomaly"]
}

dateField = {
    "oslo": "TimeWindow",
    "oslomobilities": "interval_start_time",
    "pd_cars": "start_time",
    "pd_pstrians": "start_time"
}

for dataset_name in datasets:
    numExecutions = numFoldsDict[dataset_name]
    columns = datasetSchema[dataset_name]
    date = dateField[dataset_name]

    metrics_ABOD_weighted = []
    metrics_ABOD_micro = []
    metrics_ABOD_macro = []
    CF_summary = []

    for i in range(1, numExecutions + 1):

        filename_data_train = "data/" + dataset_name + "/" + dataset_name + "_train" + str(i) + ".csv"
        filename_data_pred = "data/" + dataset_name + "/" + dataset_name + "_test" + str(i) + ".csv"

        data_train_full = pd.read_csv(filename_data_train,
                                      parse_dates=[date])

        # data_train_full.fillna(0, inplace=True)

        if (dataset_name == 'pd_cars'):
            data_train_full = data_train_full.loc[:, columns].drop(columns=[date, 'cameraname'])
        else:
            data_train_full = data_train_full.loc[:, columns].drop(columns=[date])

        # ________________________________________________________
        data_pred_full = pd.read_csv(filename_data_pred,
                                     parse_dates=[date])

        if (dataset_name == 'pd_cars'):
            data_pred_full = data_pred_full.loc[:, columns].drop(columns=[date, 'cameraname'])
            print(data_pred_full.head())
        else:
            data_pred_full = data_pred_full.loc[:, columns].drop(columns=[date])

        setting_name = dataset_name + "_ABOD"

        # data_pred_full.fillna(0, inplace=True)

        data_gt = data_pred_full.loc[:, 'anomaly']
        real_classes = np.array(data_gt).flatten()

        if not os.path.exists('logs/' + setting_name):
            os.makedirs('logs/' + setting_name)

        list_of_rows_ABOD = []
        pred_real_ABOD = []

        print("Training iteration: " + str(i))

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler_fit = min_max_scaler.fit(data_train_full)
        features = len(columns)

        ABOD_model = clf.fit(data_train_full)

        print("ABOD trained")
        print("Predicting on test")
        ABOD_preds = ABOD_model.predict(data_pred_full)
        ABOD_preds = np.where(ABOD_preds == 1, 0, ABOD_preds)
        ABOD_preds = np.where(ABOD_preds == -1, 1, ABOD_preds)

        print(np.shape(ABOD_preds))
        print(np.shape(real_classes))
        print(real_classes)
        print(ABOD_preds)

        confusion_matrix_ABOD = confusion_matrix(real_classes, ABOD_preds, labels=[0, 1])
        print("Confusion matrix:")
        print(confusion_matrix_ABOD)

        try:
            ABOD_tn, ABOD_fp, ABOD_fn, ABOD_tp = confusion_matrix_ABOD.ravel()
        except:
            ABOD_tn, ABOD_fp, ABOD_fn, ABOD_tp = [ABOD_tn, 0, 0, 0]

        print("Anomaly detection with ABOD model:")

        [precision_ABOD_weighted, recall_ABOD_weighted, fscore_ABOD_weighted,
         support_ABOD_weighted] = precision_recall_fscore_support(real_classes, ABOD_preds, average='weighted')
        [precision_ABOD_micro, recall_ABOD_micro, fscore_ABOD_micro, support_ABOD_micro] = precision_recall_fscore_support(
            real_classes, ABOD_preds, average='micro')
        [precision_ABOD_macro, recall_ABOD_macro, fscore_ABOD_macro, support_ABOD_macro] = precision_recall_fscore_support(
            real_classes, ABOD_preds, average='macro')

        metrics_ABOD_weighted.append(str(i) + "," +
            str(precision_ABOD_weighted) + "," + str(recall_ABOD_weighted) + "," + str(fscore_ABOD_weighted))
        metrics_ABOD_micro.append(str(i) + "," + str(precision_ABOD_micro) + "," + str(recall_ABOD_micro) + "," + str(fscore_ABOD_micro))
        metrics_ABOD_macro.append(str(i) + "," + str(precision_ABOD_macro) + "," + str(recall_ABOD_macro) + "," + str(fscore_ABOD_macro))

        row_if = [ABOD_tn, ABOD_fp, ABOD_fn, ABOD_tp]
        row_if_tab = [i, ABOD_tn, ABOD_fp, ABOD_fn, ABOD_tp]
        CF_summary.append(row_if_tab)

        print("TN: ", ABOD_tn, " - FP: ", ABOD_fp, " - FN: ", ABOD_fn, " - TP: ", ABOD_tp)

        list_of_rows_ABOD.append(row_if)

        # Saving predicted and real classes
        pred_real_ABOD = np.hstack((np.reshape(ABOD_preds, (np.shape(ABOD_preds)[0], 1)),
                                  np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

        np.savetxt('logs/' + setting_name + "/ABOD.preds_" + str(i) + ".log", pred_real_ABOD, delimiter=',', fmt='%s')

        # Saving confusion matrices
        import csv

        with open('logs/' + setting_name + "/ABOD.CF_summary.csv", "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(CF_summary)

        # Saving metrics
        np.savetxt('logs/' + setting_name + "/ABOD.metrics.weighted.log", metrics_ABOD_weighted,
                   delimiter=',', fmt='%s')
        np.savetxt('logs/' + setting_name + "/ABOD.metrics.macro.log", metrics_ABOD_macro, delimiter=',',
                   fmt='%s')
        np.savetxt('logs/' + setting_name + "/ABOD.metrics.micro.log", metrics_ABOD_micro, delimiter=',',
                   fmt='%s')


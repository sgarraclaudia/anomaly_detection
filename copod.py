import os
import numpy as np
import pandas as pd
import sys
from sklearn.svm import OneClassSVM
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from pyod.models.copod import COPOD

np.set_printoptions(threshold=sys.maxsize)

datasets = ["oslo"]
#datasets = ["oslo", "oslomobilities", "pd_cars","pd_pstrians"]

clf = COPOD()

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

    metrics_COPOD_weighted = []
    metrics_COPOD_micro = []
    metrics_COPOD_macro = []
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

        setting_name = dataset_name + "_COPOD"

        # data_pred_full.fillna(0, inplace=True)

        data_gt = data_pred_full.loc[:, 'anomaly']
        real_classes = np.array(data_gt).flatten()

        if not os.path.exists('logs/' + setting_name):
            os.makedirs('logs/' + setting_name)

        list_of_rows_COPOD = []
        pred_real_COPOD = []

        print("Training iteration: " + str(i))

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler_fit = min_max_scaler.fit(data_train_full)
        features = len(columns)

        COPOD_model = clf.fit(data_train_full)

        print("COPOD trained")
        print("Predicting on test")
        COPOD_preds = COPOD_model.predict(data_pred_full)
        COPOD_preds = np.where(COPOD_preds == 1, 0, COPOD_preds)
        COPOD_preds = np.where(COPOD_preds == -1, 1, COPOD_preds)

        print(np.shape(COPOD_preds))
        print(np.shape(real_classes))
        print(real_classes)
        print(COPOD_preds)

        confusion_matrix_COPOD = confusion_matrix(real_classes, COPOD_preds, labels=[0, 1])
        print("Confusion matrix:")
        print(confusion_matrix_COPOD)

        try:
            COPOD_tn, COPOD_fp, COPOD_fn, COPOD_tp = confusion_matrix_COPOD.ravel()
        except:
            COPOD_tn, COPOD_fp, COPOD_fn, COPOD_tp = [COPOD_tn, 0, 0, 0]

        print("Anomaly detection with COPOD model:")

        [precision_COPOD_weighted, recall_COPOD_weighted, fscore_COPOD_weighted,
         support_COPOD_weighted] = precision_recall_fscore_support(real_classes, COPOD_preds, average='weighted')
        [precision_COPOD_micro, recall_COPOD_micro, fscore_COPOD_micro, support_COPOD_micro] = precision_recall_fscore_support(
            real_classes, COPOD_preds, average='micro')
        [precision_COPOD_macro, recall_COPOD_macro, fscore_COPOD_macro, support_COPOD_macro] = precision_recall_fscore_support(
            real_classes, COPOD_preds, average='macro')

        metrics_COPOD_weighted.append(str(i) + "," +
            str(precision_COPOD_weighted) + "," + str(recall_COPOD_weighted) + "," + str(fscore_COPOD_weighted))
        metrics_COPOD_micro.append(str(i) + "," + str(precision_COPOD_micro) + "," + str(recall_COPOD_micro) + "," + str(fscore_COPOD_micro))
        metrics_COPOD_macro.append(str(i) + "," + str(precision_COPOD_macro) + "," + str(recall_COPOD_macro) + "," + str(fscore_COPOD_macro))

        row_if = [COPOD_tn, COPOD_fp, COPOD_fn, COPOD_tp]
        row_if_tab = [i, COPOD_tn, COPOD_fp, COPOD_fn, COPOD_tp]
        CF_summary.append(row_if_tab)

        print("TN: ", COPOD_tn, " - FP: ", COPOD_fp, " - FN: ", COPOD_fn, " - TP: ", COPOD_tp)

        list_of_rows_COPOD.append(row_if)

        # Saving predicted and real classes
        pred_real_COPOD = np.hstack((np.reshape(COPOD_preds, (np.shape(COPOD_preds)[0], 1)),
                                  np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

        np.savetxt('logs/' + setting_name + "/COPOD.preds_" + str(i) + ".log", pred_real_COPOD, delimiter=',', fmt='%s')

        # Saving confusion matrices
        import csv

        with open('logs/' + setting_name + "/COPOD.CF_summary.csv", "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(CF_summary)

        # Saving metrics
        np.savetxt('logs/' + setting_name + "/COPOD.metrics.weighted.log", metrics_COPOD_weighted,
                   delimiter=',', fmt='%s')
        np.savetxt('logs/' + setting_name + "/COPOD.metrics.macro.log", metrics_COPOD_macro, delimiter=',',
                   fmt='%s')
        np.savetxt('logs/' + setting_name + "/COPOD.metrics.micro.log", metrics_COPOD_micro, delimiter=',',
                   fmt='%s')


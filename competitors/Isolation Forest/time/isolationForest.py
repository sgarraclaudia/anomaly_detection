import os
import time
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

n_clusters = [5, 10, 25, 50, 100]
num_estimators = [10, 20, 50]

columns = ["date","year","month","day","hour","minute","weekDay", "holiday",
               "ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
               "DestinationAimedArrivalTime","OriginAimedDepartureTime","HeadwayService_False","anomaly"]

for n in n_clusters:
    dataset_name = "TF_aggr_" + str(n)
    filename_data_train = "Datasets/time_features/" + dataset_name + "_train.csv"
    filename_data_pred = "Datasets/time_features/" + dataset_name + "_test.csv"

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

    for n_estim in num_estimators:
        isolation_forest = IsolationForest(random_state=0, n_estimators=n_estim)

        setting_name = dataset_name + "_" +  "_IF_num_estimators_" + str(n_estim)

        if not os.path.exists('logs/' + setting_name):
            os.makedirs('logs/' + setting_name)

        metrics_if_weighted = []
        metrics_if_micro = []
        metrics_if_macro = []

        list_of_rows_if = []
        pred_real_if = []

        train_data = data_train_full.copy()
        train_data = train_data.drop(['date'], axis=1)
        train_data = train_data.drop(['year'], axis=1)
        print("Selected training data length: " + str(len(train_data)))

        data_pred = data_pred_full.copy()
        data_pred = data_pred.drop(['date'], axis=1)
        data_pred = data_pred.drop(['year'], axis=1)
        print(np.shape(data_pred))

        real_classes = data_gt.copy()
        real_classes = np.array(real_classes).flatten()

        # *** ISOLATION FOREST ***
        start = time.time()
        isolation_forest_model = isolation_forest.fit(train_data)
        stop = time.time()
        print(f"Training time: {stop - start}s")
        print("Isolation Forest trained")

        isolation_forest_predictions = isolation_forest.predict(data_pred)
        isolation_forest_predictions = np.where(isolation_forest_predictions == 1, 0.0, isolation_forest_predictions)
        isolation_forest_predictions = np.where(isolation_forest_predictions == -1, 1.0, isolation_forest_predictions) 

        pred_real_isolation_forest = np.hstack(
                    (np.reshape(isolation_forest_predictions, (np.shape(isolation_forest_predictions)[0], 1)),
                    np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

        confusion_matrix_if = confusion_matrix(real_classes, isolation_forest_predictions)
        if_tn, if_fp, if_fn, if_tp = confusion_matrix_if.ravel()

        [precision_if_weighted, recall_if_weighted, fscore_if_weighted, support_if_weighted] = precision_recall_fscore_support(real_classes, isolation_forest_predictions, average='weighted')
        [precision_if_micro, recall_if_micro, fscore_if_micro, support_if_micro] = precision_recall_fscore_support(real_classes, isolation_forest_predictions, average='micro')
        [precision_if_macro, recall_if_macro, fscore_if_macro, support_if_macro] = precision_recall_fscore_support(real_classes, isolation_forest_predictions, average='macro')

        print("Anomaly detection with Isolation Forest model..")
        metrics_if_weighted.append(str(precision_if_weighted) + "," + str(recall_if_weighted) + "," + str(fscore_if_weighted))
        metrics_if_micro.append(str(precision_if_micro) + "," + str(recall_if_micro) + "," + str(fscore_if_micro))
        metrics_if_macro.append(str(precision_if_macro) + "," + str(recall_if_macro) + "," + str(fscore_if_macro))

        row_if = [if_tn, if_fp, if_fn, if_tp]

        list_of_rows_if.append(row_if)

        # Saving predicted and real classes
        pred_real_if = np.hstack((np.reshape(isolation_forest_predictions, (np.shape(isolation_forest_predictions)[0], 1)),
                                np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

        np.savetxt('logs/' + setting_name + "/if.preds_" + ".log", pred_real_if, delimiter=',', fmt='%s')

        if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

        # Saving confusion matrices
        import csv

        with open('logs/' + setting_name + '/if.matrix.csv', "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(list_of_rows_if)

        # Saving predicted and real classes
        np.savetxt('logs/' + setting_name + "/if.preds.log", pred_real_isolation_forest, delimiter=',', fmt='%s')

        # Saving metrics
        np.savetxt('logs/' + setting_name + "/if.metrics.weighted.log", metrics_if_weighted, delimiter=',', fmt='%s')
        np.savetxt('logs/' + setting_name + "/if.metrics.macro.log", metrics_if_macro, delimiter=',', fmt='%s')
        np.savetxt('logs/' + setting_name + "/if.metrics.micro.log", metrics_if_micro, delimiter=',', fmt='%s')


        #sys.exit(1)
import os
import time
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from pyod.models.copod import COPOD

np.set_printoptions(threshold=sys.maxsize)

n_clusters = [5, 10, 25, 50, 100]

columns = ["date", "year", "month", "day", "hour", "minute", "weekDay", "holiday","ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
           "DestinationAimedArrivalTime","OriginAimedDepartureTime","HeadwayService_False","anomaly"]

for n in n_clusters:

    dataset_name = "TF_aggr_" + str(n)

    filename_data_train = "Datasets/" + dataset_name + "_train.csv"
    filename_data_pred = "Datasets/" + dataset_name + "_test.csv"

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

    clf = COPOD()

    setting_name = dataset_name

    if not os.path.exists('competitors/COPOD/time/logs/' + setting_name):
        os.makedirs('competitors/COPOD/time/logs/' + setting_name)

    metrics_copod_weighted = []
    metrics_copod_micro = []
    metrics_copod_macro = []

    list_of_rows_copod = []
    pred_real_copod = []

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
    copod_model = clf.fit(train_data)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    print("COPOD trained")

    copod_predictions = clf.predict(data_pred)
    copod_predictions = np.where(copod_predictions == 0, 0.0, copod_predictions)
    copod_predictions = np.where(copod_predictions == 1, 1.0, copod_predictions) 

    pred_real_copod = np.hstack(
                (np.reshape(copod_predictions, (np.shape(copod_predictions)[0], 1)),
                np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

    confusion_matrix_copod = confusion_matrix(real_classes, copod_predictions)
    copod_tn, copod_fp, copod_fn, copod_tp = confusion_matrix_copod.ravel()

    [precision_copod_weighted, recall_copod_weighted, fscore_copod_weighted, support_copod_weighted] = precision_recall_fscore_support(real_classes, copod_predictions, average='weighted')
    [precision_copod_micro, recall_copod_micro, fscore_copod_micro, support_copod_micro] = precision_recall_fscore_support(real_classes, copod_predictions, average='micro')
    [precision_copod_macro, recall_copod_macro, fscore_copod_macro, support_copod_macro] = precision_recall_fscore_support(real_classes, copod_predictions, average='macro')

    print("Anomaly detection with Copula-Based Outlier Detection model..")
    metrics_copod_weighted.append(str(precision_copod_weighted) + "," + str(recall_copod_weighted) + "," + str(fscore_copod_weighted))
    metrics_copod_micro.append(str(precision_copod_micro) + "," + str(recall_copod_micro) + "," + str(fscore_copod_micro))
    metrics_copod_macro.append(str(precision_copod_macro) + "," + str(recall_copod_macro) + "," + str(fscore_copod_macro))

    row_copod = [copod_tn, copod_fp, copod_fn, copod_tp]

    list_of_rows_copod.append(row_copod)

    # Saving predicted and real classes
    pred_real_copod = np.hstack((np.reshape(copod_predictions, (np.shape(copod_predictions)[0], 1)),
                            np.reshape(real_classes, (np.shape(real_classes)[0], 1))))

    np.savetxt('competitors/COPOD/time/logs/' + setting_name + "/if.preds_" + ".log", pred_real_copod, delimiter=',', fmt='%s')

    if not os.path.exists('competitors/COPOD/time/logs/' + setting_name):
            os.makedirs('competitors/COPOD/time/logs/' + setting_name)

    # Saving confusion matrices
    import csv

    with open('competitors/COPOD/time/logs/' + setting_name + '/if.matrix.csv', "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_rows_copod)

    # Saving predicted and real classes
    np.savetxt('competitors/COPOD/time/logs/' + setting_name + "/if.preds.log", pred_real_copod, delimiter=',', fmt='%s')

    # Saving metrics
    np.savetxt('competitors/COPOD/time/logs/' + setting_name + "/if.metrics.weighted.log", metrics_copod_weighted, delimiter=',', fmt='%s')
    np.savetxt('competitors/COPOD/time/logs/' + setting_name + "/if.metrics.macro.log", metrics_copod_macro, delimiter=',', fmt='%s')
    np.savetxt('competitors/COPOD/time/logs/' + setting_name + "/if.metrics.micro.log", metrics_copod_micro, delimiter=',', fmt='%s')


    #sys.exit(1)
import os
import numpy as np
import pandas as pd
import sys
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from pyod.models.abod import ABOD

np.set_printoptions(threshold=sys.maxsize)


n_clusters = [100]
num_estimators = [10, 20, 50]

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

    clf = ABOD()


    setting_name = dataset_name + "_ABOD"

    if not os.path.exists('competitors/ABOD/time/logs' + setting_name):
        os.makedirs('competitors/ABOD/time/logs/' + setting_name)

    metrics_ABOD_weighted = []
    metrics_ABOD_micro = []
    metrics_ABOD_macro = []

    list_of_rows_ABOD = []
    pred_real_ABOD = []

    
    train_data = data_train_full.copy()
    train_data = train_data.drop(['date'], axis=1)
    train_data = train_data.drop(['year'], axis=1)
    print("Selected training data length: " + str(len(train_data)))

    data_pred = data_pred_full.copy()
    data_pred = data_pred.drop(['date'], axis=1)
    data_pred = data_pred.drop(['year'], axis=1)
    print(np.shape(data_pred))

    # ground truth (anomaly yes/ no)
    real_classes = data_gt.copy()
    real_classes = np.array(real_classes).flatten()

    start = time.time()
    ABOD_model = clf.fit(train_data)
    print("ABOD model trained")
    stop = time.time()
    print(f"Training time: {stop - start}s")

    ABOD_model_predictions = ABOD_model.predict(data_pred)
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

    np.savetxt('competitors/ABOD/time/logs/' + setting_name + "/ABOD.preds"  + ".log", pred_real_if, delimiter=',', fmt='%s')
    

    if not os.path.exists('competitors/ABOD/time/logs/' + setting_name):
        os.makedirs('competitors/ABOD/time/logs/' + setting_name)

    # Saving confusion matrices
    import csv

    with open('competitors/ABOD/time/logs/' + setting_name + '/ABOD.matrix.csv', "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_rows_ABOD)

    # Saving predicted and real classes
    np.savetxt('competitors/ABOD/time/logs/' + setting_name + "/ABOD.preds.log", pred_real_ABOD, delimiter=',', fmt='%s')

    # Saving metrics
    np.savetxt('competitors/ABOD/time/logs/' + setting_name + "/ABOD.metrics.weighted.log", metrics_ABOD_weighted, delimiter=',', fmt='%s')
    np.savetxt('competitors/ABOD/time/logs/' + setting_name + "/ABOD.metrics.macro.log", metrics_ABOD_macro, delimiter=',', fmt='%s')
    np.savetxt('competitors/ABOD/time/logs/' + setting_name + "/ABOD.metrics.micro.log", metrics_ABOD_micro, delimiter=',', fmt='%s')


    #sys.exit(1)
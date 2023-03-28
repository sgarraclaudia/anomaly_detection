import os
import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

n_clusters = [5, 10, 25, 50, 100]

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
    
    n_neig = [10, 20, 40]
    novelty = [True]

    for neig in n_neig:
        for nov in novelty:
            local_outlier_factor = LocalOutlierFactor(n_neighbors=neig, novelty=nov)


            setting_name = dataset_name + "_LOF_num_neig_" + str(neig) + "_novelty_" + str(nov)

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            metrics_lof_weighted = []
            metrics_lof_micro = []
            metrics_lof_macro = []

            list_of_rows_lof = []
            pred_real_lof = []

            
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
            local_outlier_factor_model = local_outlier_factor.fit(train_data)
            stop = time.time()
            print(f"Training time: {stop - start}s")
            print("Local outlier factor trained")

            lof_predictions = local_outlier_factor_model.predict(data_pred)
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

            np.savetxt('logs/' + setting_name + "/lof.preds" + ".log", pred_real_lof, delimiter=',', fmt='%s')
            

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

import os
import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

n_clusters = [5, 10, 25, 50, 100]
nu_parameter = [0.1, 0.5, 1.0]
kernel = ["linear","rbf"]
gamma = 'auto'

columns = ["date","ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
           "DestinationAimedArrivalTime","OriginAimedDepartureTime","HeadwayService_False","anomaly"]

for k in kernel:
    for nu in nu_parameter:
        for n in n_clusters:
            dataset_name = "aggr_" + str(n)
            filename_data_train = "Datasets/" + dataset_name + ".txt.red.txt"
            filename_data_pred = "Datasets/" + dataset_name + ".txt.red_test.txt"

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

            one_class_svm = OneClassSVM(nu=nu, kernel=k, gamma=gamma)

            setting_name = dataset_name + "_OCSVM_nu_" + str(nu) + "_kernel_" + str(k)

            if not os.path.exists('logs/' + setting_name):
                os.makedirs('logs/' + setting_name)

            metrics_ocsvm_weighted = []
            metrics_ocsvm_micro = []
            metrics_ocsvm_macro = []

            list_of_rows_ocsvm = []
            pred_real_ocsvm = []


            train_data = data_train_full.drop(['date'], axis=1)
            print("Selected training data length: " + str(len(train_data)))

            # current day data for prediction
            data_pred = data_pred_full.drop(['date'], axis=1)
            print(np.shape(data_pred))

            # ground truth (anomaly yes/ no)
            real_classes = data_gt.copy()
            real_classes = np.array(real_classes).flatten()

            start = time.time()
            ocsvm_model = one_class_svm.fit(train_data)
            stop = time.time()
            print(f"Training time: {stop - start}s")
            print("OCSVM trained")

            ocsvm_predictions = ocsvm_model.predict(data_pred)
            ocsvm_predictions = np.where(ocsvm_predictions == 1, 0.0, ocsvm_predictions)
            ocsvm_predictions = np.where(ocsvm_predictions == -1, 1.0, ocsvm_predictions)

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

            np.savetxt('logs/' + setting_name + "/ocsvm.preds_" + ".log", pred_real_ocsvm, delimiter=',', fmt='%s')
            

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

import os
import numpy as np
import pandas as pd
import sys
import time

np.set_printoptions(threshold=sys.maxsize)

n_clusters = [5, 10, 25, 50, 100] 

columns = ["date","ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
           "DestinationAimedArrivalTime","OriginAimedDepartureTime","HeadwayService_False","anomaly"]

for n in n_clusters:
    dataset_name = "aggr_" + str(n)

    filename_data_train = "Datasets/" + dataset_name + ".txt.red.txt"
    filename_data_pred = "Datasets/" + dataset_name + ".txt.red_test.txt"

    data_train_full = pd.read_csv(filename_data_train,
                parse_dates=['date'],
                na_values=[0.0])

    data_train_full.fillna(0, inplace=True)

    data_train_full = data_train_full.loc[:, columns]

    # split the "date" column into "year", "month", "day", "hour", "minute", "second" and add the new columns to the dataframe
    data_train_full['year'] = data_train_full['date'].dt.year
    data_train_full['month'] = data_train_full['date'].dt.month
    data_train_full['day'] = data_train_full['date'].dt.day
    data_train_full['hour'] = data_train_full['date'].dt.hour
    data_train_full['minute'] = data_train_full['date'].dt.minute
    data_train_full['second'] = data_train_full['date'].dt.second
    #________________________________________________________
    data_pred_full = pd.read_csv(filename_data_pred,
                parse_dates=['date'],
                na_values=[0.0])

    data_pred_full.fillna(0, inplace=True)

    ground_truth_cols = ['anomaly']

    data_gt = data_pred_full.loc[:, ground_truth_cols]

    data_pred_full = data_pred_full.loc[:, columns]

    # split the "date" column into "year", "month", "day", "hour", "minute", "second" and add the new columns to the dataframe
    data_pred_full['year'] = data_pred_full['date'].dt.year
    data_pred_full['month'] = data_pred_full['date'].dt.month
    data_pred_full['day'] = data_pred_full['date'].dt.day
    data_pred_full['hour'] = data_pred_full['date'].dt.hour
    data_pred_full['minute'] = data_pred_full['date'].dt.minute
    data_pred_full['second'] = data_pred_full['date'].dt.second
    #________________________________________________________

    # remove the "date" column from both data_train_full and data_pred_full
    data_train_full.drop(['date'], axis=1, inplace=True)
    data_pred_full.drop(['date'], axis=1, inplace=True)

    # save the two datafreames to csv files 
    data_train_full.to_csv("Datasets/" + "ST_" + dataset_name + ".txt.red.csv", index=False)
    data_pred_full.to_csv("Datasets/" + "ST_" + dataset_name + ".txt.red_test.csv", index=False)


    
    
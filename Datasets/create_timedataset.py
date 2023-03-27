import numpy as np
import pandas as pd
import sys
import holidays
import datetime


norway_holidays = holidays.country_holidays('NO')
np.set_printoptions(threshold=sys.maxsize)

n_clusters = [5, 10, 25, 50, 100]

columns = ["date","ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
           "DestinationAimedArrivalTime","OriginAimedDepartureTime","HeadwayService_False","anomaly"]

new_columns = ["date","year","month","day","hour","minute","weekDay", "holiday",
               "ClusterLatitude","ClusterLongitude","Delay","Percentage","InPanic","InCongestion",
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

    
    data_train_full['weekDay'] =  data_train_full['date'].dt.dayofweek
    # add a new column "holiday" to the dataframe and set the value to 1 if the date is a holiday, otherwise set the value to 0
    for index, row in data_train_full.iterrows():
        if datetime.date(row['year'], row['month'], row['day']) in norway_holidays:
            data_train_full.at[index, 'holiday'] = 1
        else:
            data_train_full.at[index, 'holiday'] = 0


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
    
    data_pred_full['weekDay'] =  data_pred_full['date'].dt.dayofweek
    # add a new column "holiday" to the dataframe and set the value to 1 if the date is a holiday, otherwise set the value to 0
    for index, row in data_pred_full.iterrows():
        if datetime.date(row['year'], row['month'], row['day']) in norway_holidays:
            data_pred_full.at[index, 'holiday'] = 1
        else:
            data_pred_full.at[index, 'holiday'] = 0
    #________________________________________________________

    # create new data_train_full and data_pred_full with the new columns
    new_data_train_full = data_train_full.loc[:, new_columns]
    new_data_pred_full = data_pred_full.loc[:, new_columns]

    # copy the data_train_full and data_pred_full to new variables with the new columns
    for col in new_columns:
        new_data_train_full[col] = data_train_full[col]
        new_data_pred_full[col] = data_pred_full[col]
    
    # save the new data_train_full and data_pred_full to csv files
    new_data_train_full.to_csv("Datasets/" + "TF_" + dataset_name + "_train.csv", index=False)
    new_data_pred_full.to_csv("Datasets/" + "TF_" + dataset_name + "_test.csv", index=False)
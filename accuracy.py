import numpy as np
import os
import pandas as pd
import holidays
import datetime

norway_holidays = holidays.country_holidays('NO')

n_clusters = [5, 10, 25, 50, 100]

dataset_folder = 'Datasets/'

path_folder = 'Competitors/ABOD/logs/'
model_name = 'ABOD'
dataset_prrefix = 'aggr_'

columns_newfile = ["date", "prediction", "realValue", "actual_accuracy"]

# create a new pandas dataframe with the columns_newfile
df_final = pd.DataFrame(columns=columns_newfile)

# read the dataset
for cluster in n_clusters:

    # read the preds.log
    labels = pd.read_csv(path_folder + dataset_prrefix + str(cluster) + '_'+ model_name + '/' + model_name + '.preds.log', header=None, sep=',')

    dataset = pd.read_csv(dataset_folder+ dataset_prrefix + str(cluster) + '.txt.red_test.txt', sep=',', parse_dates=["date"], na_values=[0.0])
    
    total_pred = 0
    total_correct = 0
    current_accuracy = 0.0

    # iterate over the labels
    for i in range(len(labels)):
        # get the first column value from  labels
        prediction = labels.iloc[i, 0]
        # get the second column value from preds
        realValue = labels.iloc[i, 1]
        # read the date from dataset
        date = dataset["date"].iloc[i]

        total_pred += 1
        if prediction == realValue:
            total_correct += 1
        current_accuracy = total_correct / total_pred

        # add a new row to the dataframe df_final with the values date, prediction, realValue, current_accuracy
        df_final = df_final.append({'date': date, 'prediction': prediction, 'realValue': realValue, 'actual_accuracy': current_accuracy}, ignore_index=True)

    # save the dataframe df_final to a csv file
    df_final.to_csv(path_folder + dataset_prrefix + str(cluster) + '_'+ model_name + '/' +'accuracy.csv', index=False)


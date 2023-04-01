import pandas as pd
import matplotlib.pyplot as plt

nun_cluster = [5, 10, 25, 50, 100]
model_names = ['abod', 'ae', 'copod', 'if', 'lof', 'ocsvm']
dataset_prefix = 'aggr'

# create an empty dataframe having columns date and each value of model_names as columns
df_total = pd.DataFrame(columns=['date'] + model_names)

for model in model_names:
    # read the csv file for each model in its folder
    model_df = pd.read_csv('TF grafici/'+ model + '/' + "accuracy_" + dataset_prefix + str(nun_cluster[0]) + model+ '.csv', parse_dates=['date'])

    # print the column names of the model_df
    print(model_df.columns)

    # take the date column from the model_df and add it to the df_total
    df_total['date'] = model_df['date']

    df_total[model] = model_df['actual_accuracy']

# plot the df_total with the date as x axis and the model_names as y axis
# define the x axis with frequency of 1 day
df_total.plot(x='date', y=model_names, figsize=(20, 10), x_compat=True, rot=90)
plt.show()
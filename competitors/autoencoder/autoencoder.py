import numpy as np
import os
import keras
from keras import Input
from keras import Model
from keras.layers import Dense
import scipy.spatial.distance as spdist

# ******************************************************************************************
from keras.optimizers import Adam

os.environ['KERAS_BACKEND']='tensorflow'

def train_single_model(data, encoding_dim, epochs, batch_size, l_rate, min_max_scaler, features, model_id):

    print(type(data))

    features_list = np.arange(features)
    setting_folder = 'logs/'

    if not os.path.exists(setting_folder):
        os.makedirs(setting_folder)

    rec_err_folder = 'logs/rec_err/'

    if not os.path.exists(rec_err_folder):
        os.makedirs(rec_err_folder)

    #data = np.loadtxt(background_data_file, delimiter=',', usecols=features_list)

    x_train = data.astype('float32')
    x_test = data.astype('float32')
    x_train_norm = min_max_scaler.transform(x_train)
    x_test_norm = min_max_scaler.transform(x_test)

    print(type(x_train_norm))

    input = Input(shape=(data.shape[1],))
    encoded = Dense(encoding_dim, activation='sigmoid')(input)
    decoded = Dense(data.shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input, decoded)

    encoder = Model(input, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    adam = Adam(learning_rate=l_rate)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')

    autoencoder.fit(x_train_norm, x_train_norm,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    validation_data=(x_test_norm, x_test_norm),
                    verbose=0)

    encoded = encoder.predict(x_test_norm)
    decoded = decoder.predict(encoded)

    np.savetxt("encoded_post_training_" + str(0) + ".csv", encoded, delimiter=',', fmt='%f')

    mse = np.power(x_test_norm - decoded, 2)

    mse_r = []

    for i in range(0, np.shape(mse)[0]):
        mse_i = np.mean(abs(mse[i]))
        mse_r.append(mse_i)                         # one element per instance

    # save file with rec. errors
    np.savetxt("logs/rec_err/rec__m_" + str(0) + ".csv",
               mse_r,
               delimiter=',',
               fmt='%f')

    print('MSE on training instances', np.mean(mse_r), ' - std. dev. ', np.std(mse_r))

    # self_avg_distances = spdist.pdist(encoded)
    # print("Distance calculation done")

    # self_avg = np.mean(self_avg_distances)
    # self_std = np.std(self_avg_distances)

    # print("Self average distance for first embedding: " + str(self_avg))
    # print("Self std. dev for first embedding: " + str(self_std))

    # model = {'model_id': model_id, 'autoencoder': autoencoder, 'encoder': encoder, 'decoder': decoder,
    #          'mse_avg': np.mean(mse), 'mse_std': np.std(mse), 'embedding': encoded, 'avg_dist': self_avg, 'dist_stdev': self_std,
    #          'min_max_scaler': min_max_scaler}
    model = {'model_id': model_id, 'autoencoder': autoencoder, 'encoder': encoder, 'decoder': decoder,
             'mse_avg': np.mean(mse), 'mse_std': np.std(mse), 'embedding': encoded, 'min_max_scaler': min_max_scaler}

    return model


# ******************************************************************************************


def anomaly_detection(autoencoder, data, anomaly_std_factor, min_max_scaler):
    # Calculating thresholds for anomaly detection and concept drift based on rec. error on normal instances (training)
    anomaly_t = autoencoder['mse_avg'] + anomaly_std_factor * autoencoder['mse_std']
    print('Anomaly threshold', anomaly_t)

    count_anomalies = 0

    x_test = data.astype('float32')
    x_test_norm = min_max_scaler.transform(x_test)

    print(type(x_test_norm))

    encoded = autoencoder['encoder'].predict(x_test_norm)
    decoded = autoencoder['decoder'].predict(encoded)

    clean_data = []
    mse = []

    for i in range(0, np.shape(x_test_norm)[0]):
        mse_i = np.mean(abs(np.power(x_test_norm[i] - decoded[i], 2)))
        mse.append(mse_i)

    # Anomaly detection (by-instance logic)
    pred_classes = []

    for i in range(0, np.shape(mse)[0]):
        if mse[i] >= anomaly_t:
            pred_classes.append(1.0)        # anomaly
            count_anomalies = count_anomalies + 1
        else:
            pred_classes.append(0.0)        # normal
            clean_data.append(x_test_norm[i])

    clean_data = np.array(clean_data)

    print("Original data size: ", np.shape(data)[0])
    print("Anomalies detected: ", count_anomalies)
    print("Cleaned data size: ", np.shape(clean_data)[0])

    return pred_classes, clean_data


# ******************************************************************************************

def reconstruction(autoencoder, data, min_max_scaler):
    x_test = data.astype('float32')
    x_test_norm = min_max_scaler.transform(x_test)
    encoded = autoencoder['encoder'].predict(x_test_norm)
    decoded = autoencoder['decoder'].predict(encoded)

    mse = np.power(x_test_norm - decoded, 2)      # one element per feature

    mse_r = []

    for i in range(0, np.shape(mse)[0]):
        mse_i = np.mean(mse[i])
        mse_r.append(mse_i)                         # one element per instance

    return mse_r

# ******************************************************************************************

# ******************************************************************************************


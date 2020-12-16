import random
import pandas as pd
import math
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.metrics import RootMeanSquaredError, MAE, MAPE
from keras import metrics
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, LSTM
from keras.regularizers import l2
import tensorflow as tf
import numpy as np

column_prefixes = [
    "arrivals_daily_total",
    "occupancy_daily_peak",
    "weekday",
    "month",
    "holiday_name",
    "holiday_t-3",
    "holiday_t-2",
    "holiday_t-1",
    "holiday_t+0",
    "holiday_t+1",
    "holiday_t+2",
    "holiday_t+3",
    "is_working_day",
    "cloud_count",
    "air_pressure",
    "rel_hum",
    "rain_intensity",
    "snow_depth",
    "air_temp_min",
    "air_temp_max",
    "dew_point_temp",
    "visibility",
    "slip",
    "heat",
    "ekströms_visits",
    "ekströms_ratio",
    "website_visits_tays_total",
    #"website_visits_acuta_total",
    "public_events_num_of_daily_all"
]

true_false_labels = [
    "holiday_t-3",
    "holiday_t-2",
    "holiday_t-1",
    "holiday_t+0",
    "holiday_t+1",
    "holiday_t+2",
    "holiday_t+3",
    "is_working_day",
]

labels_to_encode = [
    "weekday",
    "month",
    "holiday_name",
    "slip",
    "heat"
]

#TARGET = ['occupancy', 'arrivals', 'diagnosis', 'mean_length_of_stay']

SAMPLESIZE = 60
BATCHSIZE = 64
LOSS = 'mae'
OPTIMIZER = 'adam'
DROPOUT = random.uniform(0.1, 0.3)
EPOCHS = 500
HIDDEN_UNITS = random.choice([4, 8, 16, 32, 64, 128])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

if __name__ == '__main__':
    # With cuda might bug out without the following session configs.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # Load data
    data = pd.read_pickle('data.pkl')['2015-05':'2019-9']
    # Get sum of 24h to a new column
    data['website_visits_tays_total'] = data['website_visits_tays'].resample('D').sum()
    # Include only predefined columns
    data = data[column_prefixes]
    # Take only daily data rows which have 'arrivals_daily_total'
    data = data[data['arrivals_daily_total'].notnull()]
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Boolean values as 0.0 or 1.0
    data[true_false_labels] = data[true_false_labels]*1.0
    # One hot encode
    data = pd.get_dummies(data, columns=labels_to_encode)
    # Make sure all inputs are floats
    data = data.astype(float)

    # Save modified dataset for debugging.
    data.to_csv('df_test.csv')

    output = data[['arrivals_daily_total']]

    # Split to train and test dataset
    train_input, test_input = train_test_split(data.values, shuffle=False, test_size=0.27)
    train_output, test_output = train_test_split(output.values, shuffle=False, test_size=0.27)
    print("Test output actual size: {}".format(np.shape(test_output)))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_input)

    # Minmax scale everything to 0 - 1 range.
    train_input_scaled = scaler.fit_transform(train_input)
    train_output_scaled = scaler.fit_transform(train_output)
    test_input_scaled = scaler.fit_transform(test_input)
    test_output_scaled = scaler.fit_transform(test_output)

    # Number of columns.
    n_features = data.shape[1]
    # define generators.
    generator = TimeseriesGenerator(train_input_scaled, train_output_scaled, length=SAMPLESIZE, batch_size=BATCHSIZE)
    generator_test = TimeseriesGenerator(test_input_scaled, test_output_scaled, length=SAMPLESIZE, batch_size=BATCHSIZE)

    opt = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

    # define model.
    """
    The requirements to use the cuDNN implementation are:
    1. activation == tanh
    2. recurrent_activation == sigmoid
    3. recurrent_dropout == 0
    4. unroll is False
    5. use_bias is True
    6. Inputs, if use masking, are strictly right-padded.
    7. Eager execution is enabled in the outermost context.
    """
    model = Sequential()
    model.add(LSTM(units=HIDDEN_UNITS, activation='tanh',
                   input_shape=(SAMPLESIZE, n_features), dropout=DROPOUT,
                   recurrent_activation='sigmoid', recurrent_dropout=0,
                   use_bias=True, kernel_regularizer=l2(0.00005),
                   recurrent_regularizer=l2(0.00005)))

    # model.add(LSTM(units=HIDDEN_UNITS, activation='tanh', dropout=DROPOUT,
    #                recurrent_activation='sigmoid', recurrent_dropout=0,
    #                use_bias=True, kernel_regularizer=l2(0.00005),
    #                recurrent_regularizer=l2(0.00005)))

    model.add(Dense(units=1))

    model.summary()

    model.compile(optimizer=opt, loss=LOSS, metrics=[metrics.mean_squared_error,
                                                     metrics.mean_absolute_error])

    # Train model
    history = model.fit(generator, epochs=EPOCHS,
                        verbose=1, validation_data=generator_test, shuffle=False)

    # Make predictions for train and test set.
    trainPredict = model.predict(generator)
    testPredict = model.predict(generator_test)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    y_true = test_output
    y_pred = testPredict

    # calculate root mean squared error and mape.
    trainScore = math.sqrt(mean_squared_error(train_output[:, 0][SAMPLESIZE:], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_output[:, 0][SAMPLESIZE:], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    mape = mean_absolute_percentage_error(test_output[:, 0][SAMPLESIZE:], testPredict[:, 0])
    print('MAPE: %.2f ' % (mape))
    mae = mean_absolute_error(test_output[:, 0][SAMPLESIZE:],
                                          testPredict[:, 0])
    print('MAE: %.2f ' % (mae))

    # Saving model
    model_name = 'models/LSTM_model_' + str(random.randint(1, 10000)) + '.h5'
    model.save(model_name)

    log_file = open('logfile.txt', 'a')
    print(
        'Sample amount: {}, Samplesize: {}, Batchsize: {}, Epochs: {}, Loss: {}, '
        'Dropout: {}, Optimizer: {}, Hidden units: {}, Test RMSE: {}, Test MAPE: {}, MAE: {}'.format(
                                            data.shape[0], SAMPLESIZE, BATCHSIZE,
                                            EPOCHS, LOSS, DROPOUT, OPTIMIZER,
                                            HIDDEN_UNITS, testScore, mape, mae),
        file=log_file)
    log_file.close()

    # Plotting loss, train and test sets.
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.legend()
    pyplot.figure()
    pyplot.plot(train_output[:, 0][SAMPLESIZE:], label='train_real')
    pyplot.plot(trainPredict[:, 0], label='train_predicted')
    pyplot.legend()
    pyplot.figure()
    pyplot.plot(test_output[:, 0][SAMPLESIZE:], label='test_real')
    pyplot.plot(testPredict[:, 0], label='test_predicted')
    pyplot.legend()
    pyplot.show()

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

def predict_model():
    df = yf.download(tickers=['BBYB.JK'], period='3y')
    y = df['Close'].fillna(method='ffill').values.reshape(- 1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the training sequences
    n_forecast = 1 
    n_lookback = 60

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    model = keras.models.load_model('./bbybjk_model.h5')

    # generate the multi-step forecasts
    n_future = 7
    y_future = []

    x_pred = X[-1:, :, :]  # last observed input sequence
    y_pred = Y[-1]         # last observed target value

    for i in range(n_future):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred.flatten()[0])

    # transform the forecasts back to the original scale
    y_future = np.array(y_future).reshape(-1, 1)
    y_future = scaler.inverse_transform(y_future)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = "null"

    df_future = pd.DataFrame(columns=['Date', 'Close', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_future)
    df_future['Forecast'] = y_future.flatten()
    df_future['Close'] = "null"

    # return df_future['Forecast']

    results = df_past.append(df_future).set_index('Date')

    return results.values.tolist()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_folder = 'data/'

api_key = ''
api_secret = ''

client = Client(api_key, api_secret)


def get_dataset_using_binance(crypto_pair: str, interval: str, start_date: str | None, save: bool):
    klines = client.get_historical_klines(crypto_pair, interval, start_str=start_date)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    if save:
        filename = f'{crypto_pair[:3].lower()}_{interval}.csv'
        df.to_csv(os.path.join(data_folder, filename))
    else:
        return df


def get_raw_data(path_to_data: str):
    path_to_data = os.path.join(data_folder, path_to_data)
    df = pd.read_csv(path_to_data)
    return df


def preprocess_df(df: pd.DataFrame):
    # df = df[['timestamp', 'open', 'high', 'low', 'close']]
    df = df[['timestamp', 'close']]
    # df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
    df = df.astype({'close': float})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_ready_dataframe(coin: str, interval: str, start_date: str = None, save: bool = True):
    filename = f'{coin.lower()}_{interval}.csv'
    if filename not in os.listdir(data_folder):
        df = get_dataset_using_binance(f'{coin.upper()}USDT', interval=interval, start_date=start_date, save=save)
        if save:
            df = get_raw_data(filename)
    else:
        df = get_raw_data(filename)
    df = preprocess_df(df)
    return df


def split_dataframe(df, ratio):
    split_index = int(len(df) * ratio)

    df1 = df[:split_index]
    df2 = df[split_index:]

    return df1, df2


def normalize_column(df, column_name, scaler: str):
    scaler = MinMaxScaler() if scaler == 'MinMax' else StandardScaler()
    column = df[column_name]
    scaled_column = scaler.fit_transform(column.values.reshape(-1, 1))
    df[column_name] = scaled_column
    return df, scaler


def inverse_transform_column(df, column_name, scaler):
    column = df[column_name]
    original_values = scaler.inverse_transform(column.values.reshape(-1, 1))
    df[column_name] = original_values.flatten()
    return df


def inverse_transform_values(values, scaler):
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def plot_price(dates, values):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, values)
    plt.title('Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.show()

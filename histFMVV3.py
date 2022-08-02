# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:59:01 2022

@author: Crypto
"""
import time
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import pytz
import aiohttp
import asyncio

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
incomplete_set = None
price_set = set()
price_dict = dict()
dataset = None

async def lookupFMV_Kaiko(date, token_symbol, before, after):
    global dataset
    date = str(date)[0:19] + "Z"
    price_dict_key = date + '-' + token_symbol
    if price_dict_key in price_set:
        return
    Time = datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%SZ')
    start_time = Time - timedelta(hours=before)
    start_time = str(start_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    end_time = Time + timedelta(hours=after)
    end_time = str(end_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    tx_url = "https://us.market-api.kaiko.io/v1/data/trades.v1/spot_exchange_rate/{}/usd"
    params = dict({"start_time": start_time,
                    "end_time": end_time,
                    "interval": "1m",
                    "page_size": 1000})
    headers = {'x-api-key': 'b45d6e160228c4514fd747d8ab16cd08'}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(tx_url.format(str(token_symbol).lower()), headers=headers, params=params) as response:
            text = await response.text()
            if response.status != 200:
                if response.status == 400:
                    price_set.add(price_dict_key)
                else:
                    print(response.status)
                    raise Exception("Response status is" + str(response.status))
            json_data = (json.loads(text))
            fmv = pd.DataFrame(json_data['data'])
            fmv = fmv[~fmv['price'].isna()]
            fmv['identifier'] = price_dict_key
            if dataset is None:
                dataset = fmv
            else:
                dataset = pd.concat([dataset, fmv], ignore_index=True)
            price_set.add(price_dict_key)

async def look_up_price(date, token_symbol):
    global dataset
    if dataset is None:
        raise Exception('look_up_price: Dataset is none')
    identifier = str(date) + '-' + token_symbol
    if identifier in price_dict:
        return price_dict[identifier]
    Time = datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%SZ')
    target_data = dataset[dataset['identifier'] == identifier]
    if len(target_data) == 0:
        price_dict[identifier] = (0, 0)
        return 0, 0
    target_data['datetime'] = ''
    target_data['delta'] = 0.0
    Time = Time.replace(tzinfo=pytz.utc)
    '''
    try:
    import calendar
    start_date = calendar.timegm(start_date.timetuple())
    '''
    for i in target_data.index:
        target_data.loc[i, 'datetime'] = datetime.fromtimestamp(int(target_data['timestamp'][i]) / 1000, tz=pytz.utc)
        target_data.loc[i, 'delta'] = abs(float((target_data['datetime'][i] - Time).total_seconds() / 3600))
    price = target_data['price'][target_data['delta'] == target_data['delta'].min()].to_list()[0]
    delta = target_data['delta'].min()
    if abs(delta) < 1:
        delta = str("{:.2f}".format(delta * 60)) + ' minutes'
    else:
        delta = str("{:.2f}".format(delta)) + ' hours'
    price_dict[identifier] = (price, delta)
    return price, delta

async def get_fmv_data(data, row_index):
    interval = 3
    row = data.iloc[row_index]
    incomplete_set.remove(row_index)
    symbol_list = []

#// TODO: stack datetime, debitAsset, creditAsset, txFeeAsset then remove_duplicates 

    if not pd.isna(row['debitAsset']):
        symbol_list.append(row['debitAsset'])
    if (not pd.isna(row['creditAsset'])) and row['creditAsset'] != 'USD':
        symbol_list.append(row['creditAsset'])
    if not pd.isna(row['txFeeAsset']):
        symbol_list.append(row['txFeeAsset'])

    for symbol in symbol_list:
        try:
            await lookupFMV_Kaiko(row['datetime'], symbol, interval, interval)
        except Exception as err:
            incomplete_set.add(row_index)
            # print(row)
           # print(err)


async def main(task_list):
    await asyncio.gather(*task_list)

#// TODO: keep all dataframe formatting outside of the loop, instead, apply function to all dataframe once 
async def set_price_value(data, row_index): 
    row = data.iloc[row_index]
    symbol_dict = dict()
    if not pd.isna(row['debitAsset']):
        symbol_dict['debitAsset'] = row['debitAsset']
    if (not pd.isna(row['creditAsset'])) and row['creditAsset'] != 'USD':
        symbol_dict['creditAsset'] = row['creditAsset']
    if not pd.isna(row['txFeeAsset']):
        symbol_dict['txFeeAsset'] = row['txFeeAsset']
    for asset_type in symbol_dict:
        price, _ = await look_up_price(row['datetime'], symbol_dict[asset_type])
        data.loc[row_index, asset_type + '_FMV'] = price



if __name__ == '__main__':
    # file_name = 'test_price.csv'
    file_name = 'test_download.csv'
    data = pd.read_csv(file_name)
    data['debitAsset_FMV'] = np.NAN
    data['creditAsset_FMV'] = np.NAN
    data['txFeeAsset_FMV'] = np.NAN
    start_time = time.perf_counter()
    loop = asyncio.get_event_loop()
    start = 0
    end = len(data) - 1
    incomplete_set = set([i for i in range(start, end + 1)])
    task_list = [loop.create_task(get_fmv_data(data, i)) for i in range(start, end + 1)]
    loop.run_until_complete(main(task_list))
    task_list = [loop.create_task(get_fmv_data(data, i)) for i in incomplete_set]
    loop.run_until_complete(main(task_list))
    set_value_list = [loop.create_task(set_price_value(data, i)) for i in range(start, end + 1)] 
    loop.run_until_complete(main(set_value_list))
    loop.close()
    print('total time:')
    print(time.perf_counter() - start_time)
    data.to_csv('FMV_output_v3.csv')
    print(incomplete_set)
    print(len(incomplete_set))
    print(price_dict)
    dataset.to_csv('test_output.csv')
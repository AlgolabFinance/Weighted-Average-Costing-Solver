# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:59:01 2022

@author: Crypto
"""
import time
import calendar

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import pytz
import aiohttp
import asyncio

pd.set_option('display.max_columns', None)
incomplete_set = set()
price_dict = dict()


async def lookupFMV_Kaiko(date, token_symbol, before, after):
    date = str(date)[0:19] + "Z"
    price_dict_key = date + '-' + token_symbol
    if price_dict_key in price_dict:
        return price_dict[price_dict_key]
    target_time = datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%SZ')
    start_time = target_time - timedelta(hours=before)
    start_time = str(start_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    end_time = target_time + timedelta(hours=after)
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
                    price_dict[price_dict_key] = (0, 0)
                    return 0, 0
                else:
                    print(response.status)
                    raise Exception("Response status is" + str(response.status))
            json_data = (json.loads(text))
            fmv = pd.DataFrame(json_data['data'])
            fmv = fmv[~fmv['price'].isna()]
            fmv['delta'] = 0.0
            fmv['delta'] = ((fmv['timestamp']/1000 - calendar.timegm(target_time.timetuple()))/3600).apply(abs)
            price = fmv['price'][fmv['delta'] == fmv['delta'].min()].to_list()[0]
            delta = fmv['delta'].min()
            if abs(delta) < 1:
                delta = str("{:.2f}".format(delta * 60)) + ' minutes'
            else:
                delta = str("{:.2f}".format(delta)) + ' hours'
            price_dict[price_dict_key] = (price, delta)
            return price, delta


async def main(task_list):
    await asyncio.gather(*task_list)


async def get_fmv(lookup, row_index, window=1):
    global incomplete_set
    row = lookup.iloc[row_index]
    try:
        lookup.loc[row_index, 'fmv'], _ = await lookupFMV_Kaiko(row['datetime'], row['token_symbol'], window / 2,
                                                                window / 2)
    except Exception as err:
        incomplete_set.add(row_index)
        lookup.loc[row_index, 'fmv'] = -1
        print(row['fmv_id'] + ': ' + str(err))
    else:
        incomplete_set.remove(row_index)


def execute(file_name, maximum_attempt=5):
    global incomplete_set
    attempts = 0
    default_window = 1
    default_error_window = 16
    start_time = time.perf_counter()
    data = pd.read_csv(file_name)
    data['debitAssetId'] = data['datetime'] + '-' + data['debitAsset']
    data['creditAssetId'] = data['datetime'] + '-' + data['creditAsset']
    data['txFeeAssetId'] = data['datetime'] + '-' + data['txFeeAsset']
    lookup = data[data['txType'] != 'Convert']
    lookup = pd.DataFrame([lookup['datetime'].to_list() * 3, lookup['debitAsset'].to_list() +
                           lookup['creditAsset'].to_list()
                           + lookup['txFeeAsset'].to_list()]).T.dropna().drop_duplicates()
    lookup.columns = ['datetime', 'token_symbol']
    lookup['fmv_id'] = lookup['datetime'] + '-' + lookup['token_symbol']
    lookup.reset_index(inplace=True, drop=True)
    start = 0
    end = len(lookup)
    incomplete_set = set([i for i in range(start, end)])
    loop = asyncio.get_event_loop()
    task_list = [loop.create_task(get_fmv(lookup, i, window=default_window)) for i in range(start, end)]
    loop.run_until_complete(main(task_list))
    while len(incomplete_set) > 0 and attempts < maximum_attempt:
        task_list = [loop.create_task(get_fmv(lookup, i, window=default_error_window)) for i in incomplete_set]
        loop.run_until_complete(main(task_list))
        attempts += 1
    loop.close()
    print('total time: ' + str(time.perf_counter() - start_time))
    print('Failures Remained: ' + str(len(incomplete_set)))
    start_time = time.perf_counter()
    data = pd.merge(data, lookup[['fmv_id', 'fmv']], how='left', left_on='debitAssetId', right_on='fmv_id')
    data.rename(columns={'fmv': 'debitAssetFMV'}, inplace=True)
    data.drop('fmv_id', inplace=True, axis=1)
    data = pd.merge(data, lookup[['fmv_id', 'fmv']], how='left', left_on='creditAssetId', right_on='fmv_id')
    data.rename(columns={'fmv': 'creditAssetFMV'}, inplace=True)
    data.drop('fmv_id', inplace=True, axis=1)
    data = pd.merge(data, lookup[['fmv_id', 'fmv']], how='left', left_on='txFeeAssetId', right_on='fmv_id')
    data.rename(columns={'fmv': 'txFeeAssetFMV'}, inplace=True)
    data.drop(columns=['fmv_id', 'creditAssetId', 'debitAssetId', 'txFeeAssetId'], inplace=True, axis=1)
    data.to_csv('FMV_output.csv')
    print('Time to merge data: ' + str(time.perf_counter() - start_time))

if __name__ == '__main__':
    file_name = 'test_download.csv'
    maximum_attempt = 5
    execute(file_name)
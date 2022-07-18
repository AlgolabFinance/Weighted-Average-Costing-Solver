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
import datetime
import pytz
import aiohttp
import asyncio

pd.set_option('display.max_columns', None)
incomplete_set = None
price_dict = dict()

async def lookupFMV_Kaiko(date, token_symbol, before, after):
    date = str(date)[0:19] + "Z"
    price_dict_key = date + '-' + token_symbol
    if price_dict_key in price_dict:
        return price_dict[price_dict_key]
    Time = datetime.datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%SZ')
    start_time = Time - datetime.timedelta(hours=before)
    start_time = str(start_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    end_time = Time + datetime.timedelta(hours=after)
    end_time = str(end_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    timestamp = int(datetime.datetime.timestamp(Time))
    tx_url = "https://us.market-api.kaiko.io/v1/data/trades.v1/spot_exchange_rate/" + \
             str(token_symbol).lower() + "/usd?interval=1m&page_size=1000"
    params = dict({"start_time": start_time, "end_time": end_time})
    headers = {'x-api-key': 'b45d6e160228c4514fd747d8ab16cd08'}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(tx_url, headers=headers, params=params) as response:
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
            fmv['datetime'] = ''
            fmv['delta'] = 0.0
            Time = Time.replace(tzinfo=pytz.utc)
            for i in fmv.index:
                fmv.loc[i, 'datetime'] = datetime.datetime.fromtimestamp(int(fmv['timestamp'][i]) / 1000, tz=pytz.utc)
                fmv.loc[i, 'delta'] = abs(float((fmv['datetime'][i] - Time).total_seconds() / 3600))
            price = fmv['price'][fmv['delta'] == fmv['delta'].min()].to_list()[0]
            delta = fmv['delta'].min()
            if abs(delta) < 1:
                delta = str("{:.2f}".format(delta * 60)) + ' minutes'
            else:
                delta = str("{:.2f}".format(delta)) + ' hours'
            price_dict[price_dict_key] = (price, delta)
            return price, delta


async def add_fmv(data, row_index):
    interval = 3
    row = data.iloc[row_index]
    incomplete_set.remove(row_index)
    try:
        if not pd.isna(row['debitAsset']):
            data.loc[row_index, 'debitAsset_FMV'], _ = await lookupFMV_Kaiko(row['datetime'], row['debitAsset'],
                                                                             interval,
                                                                             interval)
        if (not pd.isna(row['creditAsset'])) and row['creditAsset'] != 'USD':
            data.loc[row_index, 'creditAsset_FMV'], _ = await lookupFMV_Kaiko(row['datetime'], row['creditAsset'],
                                                                              interval,
                                                                              interval)
        if not pd.isna(row['txFeeAsset']):
            data.loc[row_index, 'txFeeAsset_FMV'], _ = await lookupFMV_Kaiko(row['datetime'], row['txFeeAsset'],
                                                                             interval,
                                                                             interval)
    except Exception as err:
        incomplete_set.add(row_index)
        print(row)
        print(err)


async def main(task_list):
    await asyncio.gather(*task_list)


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
    task_list = [loop.create_task(add_fmv(data, i)) for i in range(start, end + 1)]
    loop.run_until_complete(main(task_list))
    task_list = [loop.create_task(add_fmv(data, i)) for i in incomplete_set]
    loop.run_until_complete(main(task_list))
    loop.close()
    print('total time:')
    print(time.perf_counter() - start_time)
    data.to_csv('FMV_output.csv')
    print(incomplete_set)
    print(len(incomplete_set))

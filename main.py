import base64
import io
from io import StringIO
import pandas as pd
import time
import calendar
import json
from datetime import datetime, timedelta
import contextvars
import asyncio
import aiohttp
import streamlit as st
# from histFMVV4 import *


async def lookupFMV_Kaiko(date, token_symbol, before, after):
    date = str(date)[0:19] + "Z"
    price_dict_key = date + '-' + token_symbol
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
    global semaphore
    async with semaphore.get():
        async with aiohttp.ClientSession(headers=headers) as session:
            with session.get(tx_url.format(str(token_symbol).lower()), headers=headers, params=params) as response:
                text = await response.text()
                if response.status != 200:
                    if response.status == 400:
                        return 0, 0
                    else:
                        if response.status == 504:
                            await asyncio.sleep(0.01)
                        print(response.status)
                        raise Exception("Response status is " + str(response.status))
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
                return price, delta


async def gather(task_list):
    await asyncio.gather(*task_list)


async def get_fmv(lookup, row_index, window=1):
    global incomplete_set
    row = lookup.iloc[row_index]
    try:
        lookup.loc[row_index, 'fmv'], _ = await lookupFMV_Kaiko(row['datetime'], row['token_symbol'], window / 2,
                                                                window / 2)
    except Exception as err:
        incomplete_set.get().add(row_index)
        lookup.loc[row_index, 'fmv'] = -1
        print(row['fmv_id'] + ': ' + str(err))
        # st.write('Failure: ' + row['fmv_id'] + ': ' + str(err))
    else:
        incomplete_set.get().remove(row_index)


def execute(data, maximum_attempt=5):
    global incomplete_set
    global semaphore
    global task_list
    attempts = 0
    default_window = 1
    default_error_window = 16
    start_time = time.perf_counter()
    # data = pd.read_csv(file_name)
    #st.header('Preprocessing uploaded data...')
    data['debitAssetId'] = data['datetime'] + '-' + data['debitAsset']
    data['creditAssetId'] = data['datetime'] + '-' + data['creditAsset']
    data['txFeeAssetId'] = data['datetime'] + '-' + data['txFeeAsset']
    lookup = data[data['txType'] != 'Convert']
    lookup = pd.DataFrame([lookup['datetime'].to_list() * 3, lookup['debitAsset'].to_list() +
                           lookup['creditAsset'].to_list()
                           + lookup['txFeeAsset'].to_list()]).T.dropna().drop_duplicates()
    lookup.columns = ['datetime', 'token_symbol']
    lookup['fmv_id'] = lookup['datetime'] + '-' + lookup['token_symbol']
    lookup = lookup[lookup['token_symbol'] != 'USD']
    lookup.reset_index(inplace=True, drop=True)
    print('Number of records to look up: ' + str(len(lookup)))
    start = 0
    end = len(lookup)
    # st.write('Downloading data...')
    incomplete_set.set(set([i for i in range(start, end)]))
    semaphore.set(asyncio.Semaphore(2000))
    #loop = asyncio.get_event_loop()
    # loop = asyncio.get_running_loop()
    task_list.set([get_fmv(lookup, i, window=default_window) for i in range(start, end)])
    # loop.run_until_complete(main(task_list))
    asyncio.run(gather(task_list.get()))
    err_window = 4
    #st.header('Redoing the failed requests...')
    while len(incomplete_set.get()) > 0 and attempts < maximum_attempt:
        task_list.set([get_fmv(lookup, i, window=err_window) for i in incomplete_set.get()])
        # loop.run_until_complete(main(task_list))
        asyncio.run(gather(task_list.get()))
        attempts += 1
        err_window = min(16, err_window * 2)
    #st.header('Download finished')
    #st.header('total time: ' + str(time.perf_counter() - start_time))
    #st.header('Failures Remained: ' + str(len(incomplete_set)))
    print('total time: ' + str(time.perf_counter() - start_time))
    print('Failures Remained: ' + str(len(incomplete_set.get())))
    print(incomplete_set.get())
    for i in incomplete_set.get():
        print(lookup.iloc[i])
    #st.header('Merging Data..')
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
    print('Time to merge data: ' + str(time.perf_counter() - start_time))
    #st.header('Succeed!')
    return data


def get_table_download_link_csv(df, file_name='output'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    file_name += '.csv'
    return f'<a href="data:file/csv;base64,{b64}" download={file_name}>Download FMV.csv</a>'


def get_table_download_link_excel(df, file_name = 'output'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    towrite = io.BytesIO()
    file_excel = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0) # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    file_name += '.xlsx'
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download={file_name}>Download FMV.xlsx</a>'

def main():
    file_type = 'csv'
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    incomplete_set = contextvars.ContextVar('incomplete_set')
    semaphore = contextvars.ContextVar('semaphore')
    task_list = contextvars.ContextVar('task_list')
    # semaphore = asyncio.Semaphore(2000)
    uploaded_file = st.file_uploader("Please upload {0} file".format(file_type), type=['csv'], key='uploader')

    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        output_file_name = uploaded_file_name[0:uploaded_file_name.rfind('.')] + '_' + 'fmv'
        with st.spinner(text='In progress...'):
            uploaded_data = pd.read_csv(uploaded_file)
            st.write(uploaded_data)
            with st.sidebar:
                data = execute(uploaded_data)
            st.markdown(get_table_download_link_csv(data, output_file_name), unsafe_allow_html=True)
            st.markdown(get_table_download_link_excel(data, output_file_name), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
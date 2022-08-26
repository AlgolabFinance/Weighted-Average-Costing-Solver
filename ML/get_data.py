import os

import numpy as np
import pandas as pd
from decimal import Decimal



def preprocess_data(file_name):
    data = pd.read_excel(file_name, sheet_name='Raw')
    if 'rule' in data.columns:
        data.drop(columns='rule', inplace=True)
    ledger = pd.read_excel(file_name, sheet_name='LEDGER')
    rules = ledger[['txHash', 'rule']].drop_duplicates()
    hashes = data['txHash'].drop_duplicates()
    data['creditAmount'] = data['creditAmount'].apply(Decimal)
    data['debitAmount'] = data['debitAmount'].apply(Decimal)
    data['id'] = data.index
    data['helper'] = 1
    input = data[data['input'].isna()==False][['txHash', 'input']].drop_duplicates()
    input.drop(input[input['input'] == 'deprecated'].index, inplace=True)
    input['input'] = input['input'].str[0:10]
    input.rename(columns={'input': 'method_id'}, inplace=True)
    data['creditAsset'] = data['creditAsset'].str.strip()
    data['debitAsset'] = data['debitAsset'].str.strip()
    data = data[(data['creditAsset'] != '') | (data['debitAsset'] != '')]
    # input.to_csv('debug.csv')
    for hash in hashes:
        records = data[data['txHash'] == hash]
        methods = records['method'].dropna().tolist()
        if len(methods) > 0:
            data.loc[data['txHash'] == hash, 'method'] = methods[0]
        deposits = records[records['txType'] == 'Deposit']
        withdrawals = records[records['txType'] == 'Withdrawal']
        debit_assets = records['debitAsset'].drop_duplicates()
        credit_assets = records['creditAsset'].drop_duplicates()
        data.loc[data['txHash'] == hash, 'internal_count'] = len(records[records['classified'] == 'internal'])
        data.loc[data['txHash'] == hash, 'normal_count'] = len(records[records['classified'] == 'normal'])
        data.loc[data['txHash'] == hash, 'erc20_count'] = len(records[records['classified'] == 'Erc20'])
        data.loc[data['txHash'] == hash, 'erc721_count'] = len(records[records['classified'] == 'Erc721'])
        data.loc[data['txHash'] == hash, 'from_blackhole_count'] = len(
            records[records['memo'] == '0x0000000000000000000000000000000000000000'])
        data.loc[data['txHash'] == hash, 'to_blackhole_count'] = len(
            records[records['payee'] == '0x0000000000000000000000000000000000000000'])
        assets = set(debit_assets.to_list() + credit_assets.to_list())
        debit_asset_count = 0
        credit_asset_count = 0
        for asset in assets:
            to_be_removed = []
            balance = records[records['debitAsset'] == asset]['debitAmount'].sum() \
                      - records[records['creditAsset'] == asset]['creditAmount'].sum()
            if balance < 0:
                data.loc[(data['txHash'] == hash) & (data['creditAsset'] == asset), 'creditAmount'] = -balance
                if len(records[records['creditAsset'] == asset]) > 1:
                    to_be_removed = records[records['creditAsset'] == asset]['id'].tolist()[1:]
                to_be_removed = to_be_removed + records[records['debitAsset'] == asset]['id'].tolist()
                credit_asset_count += 1

            if balance > 0:
                data.loc[(data['txHash'] == hash) & (data['debitAsset'] == asset), 'debitAmount'] = balance
                if len(records[records['debitAsset'] == asset]) > 1:
                    to_be_removed = records[records['debitAsset'] == asset]['id'].tolist()[1:]
                to_be_removed = to_be_removed + records[records['creditAsset'] == asset]['id'].tolist()
                debit_asset_count += 1

            if balance == 0:
                data.loc[(data['txHash'] == hash) & (data['debitAsset'] == asset), 'helper'] = np.NAN
            for id in to_be_removed:
                data.loc[data['id'] == id, 'helper'] = np.NAN
        data.loc[data['txHash'] == hash, 'debit_asset_count'] = debit_asset_count
        data.loc[data['txHash'] == hash, 'credit_asset_count'] = credit_asset_count
    data.loc[data['payee'] == data['memo'], 'helper'] = np.NAN

    # get rid of txFee
    data.loc[(data['creditAmount'] == 0) & (data['txType'] == 'Withdrawal'), 'helper'] = np.NAN
    data.dropna(subset=['helper'], inplace=True)
    data.drop_duplicates(subset=['txHash'], inplace=True)
    data = pd.merge(data, rules, how='left', left_on='txHash', right_on='txHash')
    data = pd.merge(data, input, how='left', left_on='txHash', right_on='txHash')
    print(len(data))
    # data.to_csv('output.csv')
    dataset = data[['txHash', 'method', 'method_id','credit_asset_count',
                    'debit_asset_count', 'internal_count',
                    'normal_count', 'erc20_count', 'erc721_count',
                    'from_blackhole_count', 'to_blackhole_count', 'rule']]
    # dataset.to_csv('dataset.csv', index=False)
    return dataset


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    path = 'dataset/'
    file_list = os.listdir(path)
    dataset = pd.DataFrame()
    for file_name in file_list:
        file_path = path + file_name
        print(file_path)
        new_data = preprocess_data(file_path)
        dataset = pd.concat([dataset, new_data], ignore_index=True)
    dataset.to_csv('full_dataset.csv')
    # d = preprocess_data('dataset/1.xlsx')
    # d.to_csv('debug.csv')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

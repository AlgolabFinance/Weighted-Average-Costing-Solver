import collections
import json
import calendar

import numpy as np
import pandas as pd
from decimal import Decimal
import datetime as dt
import time
from datetime import timedelta

import requests
from pandas.api.types import CategoricalDtype


class CGL:
    def __init__(self):
        self.data = None
        self.movement_tracker = pd.DataFrame(
            {'datetime': [], 'tx_type': [], 'account': [], 'asset': [], 'current_balance': [], 'current_value': [],
             'value_per_unit': [],
             'amount_changed': [], 'value_changed': [], 'previous_balance': [],
             'previous_value': [], 'txHash': [], 'timestamp': [], '_id': [], 'previous_movement': []})

    def get_latest_balance(self, account, asset):
        target_account_records = self.movement_tracker[(self.movement_tracker['account'] == account)
                                                       & (self.movement_tracker['asset'] == asset)]
        amount, value_per_unit = Decimal('0'), Decimal('0')
        previous_movement_index = None
        if len(target_account_records) != 0:
            target_row = target_account_records.iloc[-1]
            amount, value_per_unit = target_row.current_balance, target_row.value_per_unit
            previous_movement_index = target_account_records.index.tolist()[-1]
        return amount, value_per_unit, previous_movement_index

    def update_movement_tracker(self, account, asset, amount_changed, value_changed, txHash,
                                id, datetime, tx_type, cgl=0, proceeds=0):
        original_amount, original_value_per_unit, previous_movement_index = self.get_latest_balance(account, asset)
        new_balance = original_amount + amount_changed

        # If the newly calculated balance is less than 0, reset the value to 0
        # if new_balance < 0:
        #     new_balance = 0
        if new_balance != 0:
            new_value_per_unit = (original_value_per_unit * original_amount + value_changed) / new_balance
        else:
            new_value_per_unit = 0
        temp_dict = dict()
        temp_dict['datetime'] = datetime
        temp_dict['tx_type'] = tx_type
        temp_dict['account'] = [account]
        temp_dict['asset'] = [asset]
        temp_dict['current_balance'] = [new_balance]
        temp_dict['current_value'] = [new_value_per_unit * new_balance]
        temp_dict['value_per_unit'] = [new_value_per_unit]
        temp_dict['amount_changed'] = [amount_changed]
        temp_dict['value_changed'] = [value_changed]
        temp_dict['previous_balance'] = [original_amount]
        temp_dict['previous_value'] = [original_value_per_unit]
        temp_dict['txHash'] = txHash
        temp_dict['timestamp'] = str(dt.datetime.now())
        temp_dict['_id'] = id
        temp_dict['previous_movement'] = str(previous_movement_index)
        temp_dict['cgl'] = cgl
        temp_dict['proceeds'] = proceeds
        self.movement_tracker = pd.concat([self.movement_tracker, pd.DataFrame(temp_dict)], ignore_index=True)

    # def find_opposite_deposit_account(self, txHash, amount, asset_type):
    #     data = self.data
    #     indexes = data[(data['txHash'] == txHash) & (data['debitAmount'] == amount) & (data['debitAsset'] == asset_type)].index.tolist()
    #     if len(indexes) != 1:
    #         raise Exception("Expected 1 corresponding deposit transaction but got " + str(len(indexes)))
    #     target_index = indexes[0]
    #     return target_index

    def read_data(self, file_name):
        self.data = pd.read_csv(file_name, dtype=str)
        # only keep pre8949 records
        self.data = self.data[(self.data['debitAccount'] != 'Transfer') & (self.data['creditAccount'] != 'Transfer')]
        self.data['creditAmount'] = self.data['creditAmount'].apply(Decimal)
        self.data['debitAmount'] = self.data['debitAmount'].apply(Decimal)
        self.data['histFMV'] = self.data['histFMV'].apply(Decimal)
        self.data['creditAssetFMV'] = self.data['creditAssetFMV'].apply(Decimal)
        self.data['debitAssetFMV'] = self.data['debitAssetFMV'].apply(Decimal)
        self.data['txFeeAssetFMV'] = self.data['txFeeAssetFMV'].apply(Decimal)
        self.data['processed'] = False
        self.data['Capital G&L'] = 0
        # cat_txType = CategoricalDtype(
        #     ['Buy', 'Deposit', 'Transfer', 'Convert', 'Trade', 'Withdrawal', 'Sell'],
        #     ordered=True
        # )
        # self.data['txType'] = self.data['txType'].astype(cat_txType)
        # self.data.sort_values(by=['datetime', 'txHash', 'txType'], inplace=True)
        # self.data.reset_index(inplace=True)
        # self.data['txType'] = self.data['txType'].astype(str)
        self.data['dr'] = np.NAN
        self.data['cr'] = np.NAN
        self.data.loc[self.data['txType'].isin(['Trade', 'Convert', 'Transfer']), 'dr'] = self.data['datetime'] \
                                                                                          + self.data['debitAccount'] + \
                                                                                          self.data['debitAsset']
        self.data.loc[self.data['txType'].isin(['Trade', 'Convert', 'Transfer']), 'cr'] = self.data['datetime'] \
                                                                                          + self.data['creditAccount'] + \
                                                                                          self.data['creditAsset']
        temp_df1 = self.data['cr']
        temp_df2 = self.data['dr']
        temp_df1.columns = ['pair']
        temp_df2.columns = ['pair']
        pivot = pd.concat([temp_df1, temp_df2], ignore_index=True)
        pivot.dropna(inplace=True)
        pivot = pd.DataFrame({'pair': pivot, 'cr': 0, 'dr': 0, 'check': 0}).drop_duplicates()
        pivot.reset_index(inplace=True, drop=True)
        for i, row in pivot.iterrows():
            pivot.loc[i, 'cr'] = self.data[self.data['cr'] == pivot.loc[i, 'pair']].count()['cr']
            pivot.loc[i, 'dr'] = self.data[self.data.dr == pivot.loc[i, 'pair']].count()['dr']
        pivot.loc[(pivot.dr > 0) & (pivot.cr > 0), 'check'] = 1
        self.data = pd.merge(self.data, pivot[['pair', 'check']], how='left', left_on='dr',
                             right_on='pair').drop_duplicates()
        self.data.rename(columns={'check': 'check_dr'}, inplace=True)
        self.data = pd.merge(self.data, pivot[['pair', 'check']], how='left', left_on='cr',
                             right_on='pair').drop_duplicates()
        self.data.rename(columns={'check': 'check_cr'}, inplace=True)
        self.data.drop(['pair_x', 'pair_y'], axis=1, inplace=True)
        self.data.loc[:, 'check_dr'].fillna(0, inplace=True)
        self.data.loc[:, 'check_cr'].fillna(0, inplace=True)
        self.data['seq_no'] = ''
        self.data['order_no'] = ''
        self.data.loc[self.data['txType'] == 'Buy', 'order_no'] = '1'
        self.data.loc[self.data['txType'] == 'Deposit', 'order_no'] = '2'
        self.data.loc[self.data['txType'].isin(['Trade', 'Convert', 'Transfer']), 'order_no'] = '3'
        self.data.loc[self.data['txType'] == 'Withdrawal', 'order_no'] = '4'
        self.data.loc[self.data['txType'] == 'Sell', 'order_no'] = '5'
        self.data.loc[:, 'seq_no'] = self.data.datetime + '-' + self.data.order_no
        self.data[['check_cr', 'check_dr']] = self.data[['check_cr', 'check_dr']].astype(int)
        self.data['check_count'] = self.data['check_cr'] + self.data['check_dr']
        check_set = self.data[self.data['check_count'] == 2]
        temp_df3 = check_set['cr']
        temp_df4 = check_set['dr']
        temp_df3.columns = ['pair']
        temp_df4.columns = ['pair']
        pivot_check = pd.concat([temp_df3, temp_df4], ignore_index=True)
        pivot_check.dropna(inplace=True)
        pivot_check = pd.DataFrame({'pair': pivot_check, 'cr': 0, 'dr': 0, 'check': 0}).drop_duplicates()
        for i, row in pivot_check.iterrows():
            pivot_check.loc[i, 'cr'] = check_set[check_set['cr'] == pivot_check.loc[i, 'pair']].count()['cr']
            pivot_check.loc[i, 'dr'] = check_set[check_set['dr'] == pivot_check.loc[i, 'pair']].count()['dr']
        pivot_check.loc[(pivot_check.cr > 0) & (pivot_check.dr > 0), 'check'] = 1
        # print(pivot_check)
        self.data = pd.merge(self.data, pivot_check[['pair', 'check']], how='left', left_on='dr',
                             right_on='pair').drop_duplicates()
        self.data.loc[self.data.check == 1, 'check_dr'] = 2
        self.data.sort_values('check_dr', inplace=True, ascending=False)
        self.data.sort_values('check_cr', inplace=True, ascending=True)
        self.data.sort_values('seq_no', inplace=True)
        self.data.drop('pair', inplace=True, axis=1)
        # print(pivot)
        # print(self.data)

    def calculate_CGL(self, record):
        tx_type = record.txType
        if tx_type not in ['Trade', 'Sell']:
            raise Exception("Wrong txType: " + record['txType'])
        FMV = record.histFMV
        if FMV == 0:
            FMV = record.debitAssetFMV
        credit_account = record.creditAccount
        credit_amount = record.creditAmount
        credit_asset = record.creditAsset
        debit_account = record.debitAccount
        debit_amount = record.debitAmount
        debit_asset = record.debitAsset
        credit_asset_fmv = record.creditAssetFMV
        tx_hash = record.txHash
        id = record._id
        datetime = record.datetime
        proceeds = Decimal('0')
        cost = Decimal('0')
        if tx_type == 'Trade':
            proceeds = FMV * debit_amount
        else:
            proceeds = debit_amount
        balance_amount, value_per_unit, previous_movement_index = self.get_latest_balance(credit_account, credit_asset)
        # if credit_amount > balance_amount:
        #     raise Exception(id + " :Negative Balance for account: " + str(credit_asset) + ". Current balance: "
        #                     + str(balance_amount) + '  Credit Amount: ' + str(credit_amount))
        if credit_amount > balance_amount:
            cost = balance_amount * value_per_unit + (credit_amount - balance_amount) * credit_asset_fmv
        else:
            cost = credit_amount * value_per_unit
        CGL = proceeds - cost
        # self.add_value_to_account(credit_account, credit_asset, -1*credit_amount, -1*cost)
        self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount, -1 * cost, tx_hash, id, datetime,
                                     tx_type, CGL, proceeds)
        if tx_type == 'Trade':
            # self.add_value_to_account(debit_account, debit_asset, debit_amount, proceeds)
            self.update_movement_tracker(debit_account, debit_asset, debit_amount, proceeds, tx_hash, id, datetime,
                                         tx_type)
        return CGL

    def execute_calculation(self):
        self.data.reset_index(inplace=True, drop=True)
        index = 0
        max_index = self.data.index.to_list()[-1]

        while index <= max_index:
            record = self.data.loc[index, :].copy()
            tx_type = record.txType
            tx_hash = record.txHash
            credit_account = record.creditAccount
            credit_amount = record.creditAmount
            credit_asset = record.creditAsset
            debit_account = record.debitAccount
            debit_amount = record.debitAmount
            debit_asset = record.debitAsset
            credit_asset_fmv = record.creditAssetFMV
            FMV = record.histFMV
            id = record._id
            datetime = record.datetime
            balance, value_per_unit = 0, 0

            if tx_type not in ('Trade', 'Deposit', 'Withdrawal', 'Buy', 'Sell', 'Convert', 'Transfer'):
                raise Exception("Invalid txType: " + tx_type)

            # add offset income
            if tx_type in ('Withdrawal', 'Convert', 'Transfer', 'Trade', 'Sell'):
                balance, value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                if balance < credit_amount and credit_account in wallet_list:
                    offset_income_amount = credit_amount - balance
                    new_row = record
                    new_row['txType'] = 'Deposit'
                    new_row['creditAccount'] = 'Offset Income'
                    new_row['creditAsset'] = credit_asset
                    new_row['creditAmount'] = offset_income_amount
                    new_row['debitAccount'] = credit_account
                    new_row['debitAsset'] = credit_asset
                    new_row['debitAmount'] = offset_income_amount
                    new_row['debitAssetFMV'] = record['creditAssetFMV']
                    new_row['histFMV'] = new_row['debitAssetFMV']
                    new_row['creditAssetFMV'] = new_row['debitAssetFMV']
                    self.data.loc[index - 0.5] = new_row
                    self.data = self.data.sort_index().reset_index(drop=True)
                    max_index = self.data.index.to_list()[-1]
                    continue

            if tx_type in ['Trade', 'Sell']:
                cgl = self.calculate_CGL(record)
                self.data.loc[index, 'Capital G&L'] = cgl
            elif tx_type == 'Withdrawal':
                # balance, value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                value_changed = 0
                if balance < credit_amount:
                    value_changed = -1 * (value_per_unit * balance + (credit_amount - balance) * credit_asset_fmv)
                else:
                    value_changed = -1 * value_per_unit * credit_amount
                self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                             value_changed, tx_hash, id, datetime, tx_type)

            elif tx_type == 'Deposit':
                self.update_movement_tracker(debit_account, debit_asset, debit_amount, debit_amount * FMV
                                             , tx_hash, id, datetime, tx_type)

            elif tx_type == 'Buy':
                self.update_movement_tracker(debit_account, debit_asset, debit_amount, credit_amount
                                             , tx_hash, id, datetime, tx_type)
            elif tx_type == 'Convert':
                # balance, value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                # if balance < credit_amount:
                #     raise Exception("Negative Balance for account: " + str(debit_asset) + ". Current balance: "
                #                     + str(balance) + '  Credit Amount: ' + str(credit_amount))
                if balance < credit_amount:
                    value_changed = value_per_unit * balance + (credit_amount - balance) * credit_asset_fmv
                    self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                                 -1 * value_changed, tx_hash, id, datetime, tx_type)
                    self.update_movement_tracker(debit_account, debit_asset, debit_amount, value_changed,
                                                 tx_hash, id, datetime, tx_type)
                else:
                    self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                                 -1 * value_per_unit * credit_amount, tx_hash, id, datetime, tx_type)
                    self.update_movement_tracker(debit_account, debit_asset, debit_amount,
                                                 value_per_unit * credit_amount,
                                                 tx_hash, id, datetime, tx_type)
            elif tx_type == 'Transfer':
                # balance, value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                _, debit_value_per_unit, _ = self.get_latest_balance(debit_account, debit_asset)
                value_changed = 0
                if balance < credit_amount:
                    value_changed = value_per_unit * balance + (
                            credit_amount - balance) * credit_asset_fmv
                    # self.update_balance_tracker(credit_account, credit_asset, -1 * credit_amount,
                    #                             -1 * value_changed, tx_hash, id, datetime)
                    # self.update_balance_tracker(debit_account, debit_asset, credit_amount,
                    #                             value_changed, tx_hash, id, datetime)
                else:
                    value_changed = value_per_unit * credit_amount
                self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                             -1 * value_changed, tx_hash, id, datetime, tx_type)
                self.update_movement_tracker(debit_account, debit_asset, credit_amount,
                                             value_changed, tx_hash, id, datetime, tx_type)
            self.data.loc[index, 'processed'] = True
            index += 1

    def write_to_file(self, file_name):
        self.data.to_csv(file_name)
        self.movement_tracker.to_csv('movement_tracker.csv')

    def generate_transactions_report(self, movement_tracker_df, end_year, end_month,
                                     end_day, start_year=2001, start_month=1, start_day=1):
        # tx_report = pd.DataFrame(
        #     {'Account': [], 'Asset': [], 'Datetime': [], 'Current Balance': [],
        #      'Average Value': [], 'Total Value': [],
        #      'Balance Before Change': [], 'Total Value Before Change': [], 'Balance Changed': [],
        #      'Value Changed': []})
        # tracker_book = pd.read_csv('movement_tracker.csv')
        end_date = dt.datetime(end_year, end_month, end_day, 23, 59, 59)
        end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        start_date = dt.datetime(start_year, start_month, start_day, 0, 0, 0)
        start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        tx_report = movement_tracker_df[
            (movement_tracker_df.datetime <= end_date) & (movement_tracker_df.datetime >= start_date)]
        tx_report = tx_report[['account', 'asset', 'datetime', 'current_balance', 'value_per_unit',
                               'current_value', 'previous_balance', 'amount_changed', 'value_changed',
                               'previous_value']]
        tx_report['Total Value Before Change'] = tx_report['previous_value'] * tx_report['previous_balance']
        tx_report.drop('previous_value', axis=1, inplace=True)
        tx_report.rename(columns={'account': 'Account', 'asset': 'Asset', 'datetime': 'Datetime',
                                  'current_balance': 'Current Balance', 'value_per_unit': 'Basis Per Unit',
                                  'current_value': 'Total Value', 'previous_balance': 'Balance Before Change'
            , 'amount_changed': 'Balance Changed', 'value_changed': 'Value Changed', }, inplace=True)
        tx_report.set_index(['Account', 'Asset', 'Datetime'], inplace=True)
        tx_report.sort_index(inplace=True)
        tx_report.reset_index(inplace=True)
        return tx_report

    def generate_transactions_report_by_year(self, movement_tracker_df, start_year, end_year):
        min_datetime = movement_tracker_df.loc[0, 'datetime']
        min_year = int(min_datetime[0:4])
        max_datetime = movement_tracker_df.iloc[-1]['datetime']
        max_year = int(max_datetime[0:4])
        if end_year > max_year:
            end_year = max_year
        if start_year < min_year:
            start_year = min_year
        while start_year <= end_year:
            previous_report = self.generate_transactions_report(movement_tracker_df, start_year, 12, 31,
                                                                  start_year, 1, 1)
            previous_report.to_csv(str(start_year) + '-tx-report.csv')
            start_year += 1


    def generate_cgl_report(self, movement_tracker_df, end_year, end_month, end_day,
                                       start_year=2001, start_month=1, start_day=1):
        end_date = dt.datetime(end_year, end_month, end_day, 23, 59, 59)
        end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        start_date = dt.datetime(start_year, start_month, start_day, 0, 0, 0)
        start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        dataset = movement_tracker_df[
            (movement_tracker_df.datetime <= end_date) & (movement_tracker_df.datetime >= start_date)]
        # print(dataset)
        cgl_report = dataset[dataset['cgl'] != 0]
        if cgl_report.empty:
            return None
        cgl_report = cgl_report[['account', 'asset', 'datetime', 'previous_value', 'value_changed',
                                 'cgl', 'proceeds', 'amount_changed', '_id', 'txHash']]
        cgl_report['amount_changed'] = cgl_report['amount_changed'].apply(abs)
        cgl_report['value_changed'] = cgl_report['value_changed'].abs()
        cgl_report['proceeds_per_coin'] = cgl_report['proceeds'] / cgl_report['amount_changed']
        # cgl_report['purchase_date'] = np.NAN
        # for i, row in cgl_report.iterrows():
        #     account = row.account
        #     asset = row.asset
        #     cgl_report.loc[i, 'purchase_date'] = \
        #         cgl_book[(cgl_book['account'] == account) & (cgl_book['asset'] == asset)
        #                  & (cgl_book['previous_balance'] == 0)]['datetime'].to_list()[-1]

        cgl_report.set_index(['account', 'asset'], inplace=True)
        cgl_report.sort_index(inplace=True)
        cgl_report.reset_index(inplace=True)
        cgl_report['OPENING BALANCE'] = 0
        prior_dataset = movement_tracker_df[movement_tracker_df['datetime'] <= start_date]
        print(cgl_report)
        for i, row in cgl_report.iterrows():
            record = prior_dataset[(prior_dataset['account'] == row['account']) &
                          (prior_dataset['asset'] == row['asset'])]
            if not record.empty:
                cgl_report.loc[i, 'OPENING BALANCE'] = record.current_balance.to_list()[-1]


        cgl_report.rename(columns={'account': 'COIN LOCATION', 'asset': 'ASSET', 'datetime': 'PROCEED DATE',
                                   "amount_changed": "QUANTITY",
                                   'previous_value': 'BASIS PER COIN', 'proceeds': 'PROCEEDS', 'value_changed': 'BASIS',
                                   'proceeds_per_coin': 'PROCEEDS PER COIN', 'cgl': 'CAPITAL GAIN(LOSS)', '_id': '_ID'}
                          , inplace=True)
        cgl_report = cgl_report[
            ['COIN LOCATION', 'ASSET', 'QUANTITY', 'PROCEED DATE', 'BASIS PER COIN', 'PROCEEDS PER COIN',
             'BASIS', 'PROCEEDS', 'CAPITAL GAIN(LOSS)', 'OPENING BALANCE', '_ID']]
        return cgl_report

    def generate_cgl_report_by_year(self, movement_tracker_df, start_year, end_year):
        min_datetime = movement_tracker_df.loc[0, 'datetime']
        min_year = int(min_datetime[0:4])
        max_datetime = movement_tracker_df.iloc[-1]['datetime']
        max_year = int(max_datetime[0:4])
        if end_year > max_year:
            end_year = max_year
        if start_year < min_year:
            start_year = min_year
        while start_year <= end_year:
            report = self.generate_cgl_report(movement_tracker_df, start_year, 12, 31,
                                                                      start_year, 1, 1)
            if report is not None:
                report.to_csv(str(start_year) + '-cgl-report.csv')
            start_year += 1

    def generate_unrealized_cgl_report(self, movement_tracker_df, end_year, end_month, end_day,
                                       start_year=2001, start_month=1, start_day=1):
        end_date = dt.datetime(end_year, end_month, end_day, 23, 59, 59)
        end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        start_date = dt.datetime(start_year, start_month, start_day, 0, 0, 0)
        start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        dataset = movement_tracker_df[(movement_tracker_df.datetime <= end_date)]
        grouped = dataset.groupby(['account', 'asset'])
        latest_balances = None
        for (account, asset), group in grouped:
            if latest_balances is None:
                latest_balances = group.iloc[-1].to_frame().T
            else:
                latest_balances = pd.concat([latest_balances, group.iloc[-1].to_frame().T], ignore_index=True)
        print(latest_balances)
        unrealized_report = latest_balances[['asset', 'account', 'current_balance', 'value_per_unit', 'current_value']]
        unrealized_report = unrealized_report.sort_values(['asset'], ignore_index=True)
        unrealized_report['FMV PER UNIT'] = np.nan
        unrealized_report['TOTAL USD VALUE'] = np.nan
        unrealized_report['UNREALIZED GAIN (LOSS)'] = np.nan
        maximum_trial = 4
        trial = 0
        row_number = len(unrealized_report)
        i = 0
        price_dict = dict()
        while i < row_number:
            row = unrealized_report.iloc[i].copy()
            token_symbol = row['asset']
            print(token_symbol)
            if token_symbol in price_dict:
                unrealized_report.loc[i, 'FMV PER UNIT'] = price_dict[token_symbol]
                i += 1
                continue
            try:
                unrealized_report.loc[i, 'FMV PER UNIT'], _ = lookupFMV_Kaiko(end_date, token_symbol, 8, 8)
            except Exception as err:
                print(err)
                if trial < maximum_trial:
                    trial += 1
                    unrealized_report.loc[i, 'FMV PER UNIT'] = 0
                else:
                    trial = 0
                    i += 1
                    price_dict[token_symbol] = 0
            else:
                price_dict[token_symbol] = unrealized_report.loc[i, 'FMV PER UNIT']
                i += 1
        unrealized_report.rename(columns={'account': 'COIN LOCATION', 'asset': 'ASSET', 'current_balance': 'TOTAL BALANCE',
                                   'value_per_unit': 'BASIS PER UNIT', 'current_value': 'BASIS BALANCE'}, inplace=True)
        unrealized_report['TOTAL USD VALUE'] = unrealized_report['TOTAL BALANCE'].astype('float') * \
                                           unrealized_report['FMV PER UNIT'].astype('float')
        unrealized_report['UNREALIZED GAIN (LOSS)'] = unrealized_report['TOTAL USD VALUE'] - unrealized_report['BASIS BALANCE']
        unrealized_report['OPENING BALANCE'] = 0
        # unrealized_report['OPENING BASIS PER UNIT'] = 0
        # unrealized_report['OPENING BASIS BALANCE'] = 0
        prior_dataset = movement_tracker_df[movement_tracker_df['datetime'] <= start_date]
        for i, row in unrealized_report.iterrows():
            record = dataset[(prior_dataset['account'] == row['COIN LOCATION']) &
                          (prior_dataset['asset'] == row['ASSET'])]
            if not record.empty:
                unrealized_report.loc[i, 'OPENING BALANCE'] = record.current_balance.to_list()[-1]

        print(unrealized_report)
        return unrealized_report

    def generate_unrealized_report_by_year(self, movement_tracker_df, start_year, end_year):
        min_datetime = movement_tracker_df.loc[0, 'datetime']
        min_year = int(min_datetime[0:4])
        max_datetime = movement_tracker_df.iloc[-1]['datetime']
        max_year = int(max_datetime[0:4])
        if end_year > max_year:
            end_year = max_year
        if start_year < min_year:
            start_year = min_year
        while start_year <= end_year:
            previous_report = self.generate_unrealized_cgl_report(movement_tracker_df, start_year, 12, 31,
                                                                  start_year, 1, 1)
            previous_report.to_csv(str(start_year) + '-unrealized-report.csv')
            start_year += 1



def lookupFMV_Kaiko(date, token_symbol, before, after):
    if '-' in token_symbol:
        return 0, 0
    date = str(date)[0:19] + "Z"
    target_time = dt.datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%SZ')
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
    response = requests.get(tx_url.format(str(token_symbol).lower()), headers=headers, params=params)
    text = response.text
    if response.status_code != 200:
        if response.status_code == 400:
            return 0, 0
        else:
            raise Exception("Response status is " + str(response.status_code))
    json_data = (json.loads(text))
    fmv = pd.DataFrame(json_data['data'])
    fmv = fmv[~fmv['price'].isna()]
    fmv['delta'] = 0.0
    fmv['delta'] = ((fmv['timestamp'] / 1000 - calendar.timegm(target_time.timetuple())) / 3600).apply(abs)
    price = fmv['price'][fmv['delta'] == fmv['delta'].min()].to_list()[0]
    delta = fmv['delta'].min()
    if abs(delta) < 1:
        delta = str("{:.2f}".format(delta * 60)) + ' minutes'
    else:
        delta = str("{:.2f}".format(delta)) + ' hours'
    return price, delta


if __name__ == '__main__':
    start_time = time.perf_counter()
    pd.set_option('display.max_columns', None)
    wallet_list = ['KeeperDAO', 'CTO Wallet', 'Prop Wallet', 'CTO Wallet 3', 'Labs Wallet', 'Binance']
    cgl = CGL()
    # cgl.read_data('pre8949_fmv.csv')
    # cgl.read_data('test_abnormal.csv')
    # cgl.read_data('test_v4.csv')
    # cgl.read_data('test_neg.csv')
    # cgl.execute_calculation()
    # print(cgl.movement_tracker)
    # print(cgl.data)
    # cgl.write_to_file('result_neg.csv')
    # cgl.write_to_file('pre8949_output.csv')
    movement_tracker_df = pd.read_csv('movement_tracker.csv')
    cgl.generate_transactions_report(movement_tracker_df, 2021, 12, 31).to_csv('tx_report.csv')
    cgl.generate_transactions_report_by_year(movement_tracker_df, 2019, 2022)
    cgl.generate_cgl_report_by_year(movement_tracker_df, 2019, 2022)
    # cgl.generate_cgl_report(movement_tracker_df)
    # cgl.generate_unrealized_cgl_report(movement_tracker_df, 2021, 12, 31).to_csv('unrealized_cgl_report.csv')
    # cgl.generate_unrealized_report_by_year(movement_tracker_df, 2022, 2022)
    print(time.perf_counter() - start_time)

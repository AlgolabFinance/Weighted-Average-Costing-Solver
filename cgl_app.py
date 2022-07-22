import base64
import io

import numpy as np
import pandas as pd
from decimal import Decimal
import datetime as dt
import time
import streamlit as st


class CGL:
    def __init__(self):
        self.data = None
        self.movement_tracker = pd.DataFrame(
            {'datetime': [], 'account': [], 'asset': [], 'current_balance': [], 'current_value': [],
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
                                id, datetime, cgl=0, proceeds=0):
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
        temp_dict['datetime'] = datetime
        self.movement_tracker = pd.concat([self.movement_tracker, pd.DataFrame(temp_dict)], ignore_index=True)

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
        pivot.reset_index(inplace=True)
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

    def calculate_CGL(self, record):
        tx_type = record.txType
        if tx_type not in ['Trade', 'Sell']:
            raise Exception("Wrong txType: " + record['txType'])
        FMV = record.histFMV
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
                                    CGL, proceeds)
        if tx_type == 'Trade':
            # self.add_value_to_account(debit_account, debit_asset, debit_amount, proceeds)
            self.update_movement_tracker(debit_account, debit_asset, debit_amount, proceeds, tx_hash, id, datetime)
        return CGL

    def execute_calculation(self):
        for index, record in self.data.iterrows():
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
            if tx_type not in ('Trade', 'Deposit', 'Withdrawal', 'Buy', 'Sell', 'Convert', 'Transfer'):
                raise Exception("Invalid txType: " + tx_type)
            if tx_type in ['Trade', 'Sell']:
                cgl = self.calculate_CGL(record)
                self.data.loc[index, 'Capital G&L'] = cgl
            elif tx_type == 'Withdrawal':
                balance, value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                value_changed = 0
                if balance < credit_amount:
                    value_changed = -1 * (value_per_unit * balance + (credit_amount - balance) * credit_asset_fmv)
                else:
                    value_changed = -1 * value_per_unit * credit_amount
                self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                             value_changed, tx_hash, id, datetime)

            elif tx_type == 'Deposit':
                self.update_movement_tracker(debit_account, debit_asset, debit_amount, debit_amount * FMV
                                             , tx_hash, id, datetime)

            elif tx_type == 'Buy':
                self.update_movement_tracker(debit_account, debit_asset, debit_amount, credit_amount
                                             , tx_hash, id, datetime)
            elif tx_type == 'Convert':
                balance, value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                # if balance < credit_amount:
                #     raise Exception("Negative Balance for account: " + str(debit_asset) + ". Current balance: "
                #                     + str(balance) + '  Credit Amount: ' + str(credit_amount))
                if balance < credit_amount:
                    value_changed = value_per_unit * balance + (credit_amount - balance) * credit_asset_fmv
                    self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                                 -1 * value_changed, tx_hash, id, datetime)
                    self.update_movement_tracker(debit_account, debit_asset, debit_amount, value_changed,
                                                 tx_hash, id, datetime)
                else:
                    self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                                 -1 * value_per_unit * credit_amount, tx_hash, id, datetime)
                    self.update_movement_tracker(debit_account, debit_asset, debit_amount,
                                                 value_per_unit * credit_amount,
                                                 tx_hash, id, datetime)
            elif tx_type == 'Transfer':
                credit_balance, credit_value_per_unit, _ = self.get_latest_balance(credit_account, credit_asset)
                _, debit_value_per_unit, _ = self.get_latest_balance(debit_account, debit_asset)
                value_changed = 0
                if credit_balance < credit_amount:
                    value_changed = credit_value_per_unit * credit_balance + (
                            credit_amount - credit_balance) * Decimal(credit_asset_fmv)
                    # self.update_balance_tracker(credit_account, credit_asset, -1 * credit_amount,
                    #                             -1 * value_changed, tx_hash, id, datetime)
                    # self.update_balance_tracker(debit_account, debit_asset, credit_amount,
                    #                             value_changed, tx_hash, id, datetime)
                else:
                    value_changed = credit_value_per_unit * credit_amount
                self.update_movement_tracker(credit_account, credit_asset, -1 * credit_amount,
                                             -1 * value_changed, tx_hash, id, datetime)
                self.update_movement_tracker(debit_account, debit_asset, credit_amount,
                                             value_changed, tx_hash, id, datetime)
            self.data.loc[index, 'processed'] = True

    def write_to_file(self, file_name):
        self.data.to_csv(file_name)
        self.movement_tracker.to_csv('movement_tracker.csv')

    def generate_transactions_report(self):
        tx_report = self.movement_tracker
        tx_report = tx_report[['account', 'asset', 'datetime', 'current_balance', 'value_per_unit',
                               'current_value', 'previous_balance', 'amount_changed', 'value_changed',
                               'previous_value']]
        tx_report['Total Value Before Change'] = tx_report['previous_value'] * tx_report['previous_balance']
        tx_report.drop('previous_value', axis=1, inplace=True)
        tx_report.rename(columns={'account': 'Account', 'asset': 'Asset', 'datetime': 'Datetime',
                                  'current_balance': 'Current Balance', 'value_per_unit': 'Average Value',
                                  'current_value': 'Total Value', 'previous_balance': 'Balance Before Change'
            , 'amount_changed': 'Balance Changed', 'value_changed': 'Value Changed', }, inplace=True)
        tx_report.set_index(['Account', 'Asset', 'Datetime'], inplace=True)
        tx_report.sort_index(inplace=True)
        tx_report.reset_index(inplace=True)
        # tx_report.to_csv("tx_report.csv", index=False)
        return tx_report

    def generate_cgl_report(self, movement_tracker_df):
        cgl_report = movement_tracker_df[movement_tracker_df['cgl'] != 0]
        cgl_report = cgl_report[['account', 'asset', 'datetime', 'previous_value', 'value_changed',
                                 'cgl', 'proceeds', 'amount_changed', '_id', 'txHash']]
        cgl_report['amount_changed'] = cgl_report['amount_changed'].apply(abs)
        cgl_report['value_changed'] = cgl_report['value_changed'].abs()
        cgl_report['proceeds_per_coin'] = cgl_report['proceeds'] / cgl_report['amount_changed']
        cgl_report['purchase_date'] = np.NAN
        for i, row in cgl_report.iterrows():
            account = row.account
            asset = row.asset
            cgl_report.loc[i, 'purchase_date'] = \
            movement_tracker_df[(movement_tracker_df['account'] == account) & (movement_tracker_df['asset'] == asset)
                                & (movement_tracker_df['previous_balance'] == 0)]['datetime'].to_list()[-1]
        cgl_report.set_index(['account', 'asset'], inplace=True)
        cgl_report.sort_index(inplace=True)
        cgl_report.reset_index(inplace=True)
        cgl_report.rename(columns={'account': 'COIN LOCATION', 'asset': 'ASSET', 'datetime': 'DATETIME',
                                   'purchase_date': 'ORIGINAL PURCHASE DATE', "amount_changed": "AMOUNT",
                                   'previous_value': 'BASIS PER COIN', 'proceeds': 'PROCEEDS', 'value_changed': 'BASIS',
                                   'proceeds_per_coin': 'PROCEEDS PER COIN', 'cgl': 'CAPITAL GAIN(LOSS)'}
                          , inplace=True)
        cgl_report = cgl_report[
            ['COIN LOCATION', 'ASSET', 'DATETIME', 'ORIGINAL PURCHASE DATE', 'AMOUNT', 'BASIS PER COIN',
             'BASIS', 'PROCEEDS', 'PROCEEDS PER COIN', 'CAPITAL GAIN(LOSS)']]
        cgl_report.to_csv('cgl_report.csv', index=False)
        print(cgl_report)
        return cgl_report


def get_table_download_link_csv(df, file_name='output', description=''):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    file_name += '.csv'
    return f'<a href="data:file/csv;base64,{b64}" download={file_name}>Download {description} CSV</a>'


def get_table_download_link_excel(df, file_name='output', description=''):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    towrite = io.BytesIO()
    file_excel = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    file_name += '.xlsx'
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download={file_name}>Download {description} XLSX</a>'


if __name__ == '__main__':
    file_type = 'csv'
    pd.set_option('display.max_columns', None)
    uploaded_file = st.file_uploader("Please upload {0} file".format(file_type), type=['csv'], key='uploader')
    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        output_file_prefix = uploaded_file_name[0:uploaded_file_name.rfind('.')] + '_'
        cgl = CGL()
        with st.spinner(text='In progress...'):
            cgl.read_data(uploaded_file)
            st.header('Preprocessed Uploaded Data')
            st.write(cgl.data.astype(str))
            cgl.execute_calculation()
            tx_report = cgl.generate_transactions_report()
            st.header('Movement Tracker')
            st.write(cgl.movement_tracker.astype('str'))
            st.header('Transaction Report')
            tx_report = tx_report.astype('str')
            st.write(tx_report)
            cgl_report = cgl.generate_cgl_report(movement_tracker_df=cgl.movement_tracker)
            cgl_report = cgl_report.astype(str)
            st.header('Capital Gain & Loss Report')
            st.write(cgl_report)
            cgl.data = cgl.data.astype('str')
            cgl.movement_tracker = cgl.movement_tracker.astype('str')
        with st.sidebar:
            st.markdown(get_table_download_link_csv(cgl.data, output_file_prefix + 'processed_data', 'Processed Data'),
                        unsafe_allow_html=True)
            st.markdown(get_table_download_link_excel(cgl.movement_tracker, output_file_prefix + 'movement_tracker', 'Movement Tracker'),
                        unsafe_allow_html=True)
            st.markdown(
                get_table_download_link_excel(tx_report, output_file_prefix + 'tx_report', 'Transaction Report'),
                unsafe_allow_html=True)
            st.markdown(get_table_download_link_excel(cgl_report, output_file_prefix + 'cgl_report', 'CGL Report'),
                        unsafe_allow_html=True)

            # st.markdown(get_table_download_link_csv(data, output_file_name), unsafe_allow_html=True)
            # st.markdown(get_table_download_link_excel(data, output_file_name), unsafe_allow_html=True)

import glob
import os

import numpy as np
import pandas as pd
import calendar


def get_grouped_cgl(file_name, mapping_file):
    mapping = pd.read_excel(mapping_file, sheet_name='Wallet Directory')
    mapping = mapping[~mapping['cryptoCurrency'].isna()][['cryptoCurrency', 'quickbookAccount']]
    nft_list = mapping[(mapping['quickbookAccount'].str.contains('1155'))
                       | (mapping['quickbookAccount'].str.contains('721'))]['cryptoCurrency'].tolist()
    data = pd.read_csv(file_name)
    if data.empty:
        return None, None
    data['class'] = np.NAN
    long_term_index = data[data['Description of property'] == 'Long Term'].index.tolist()[0]
    short_term_index = data[data['Description of property'] == 'Short Term'].index.tolist()[0]
    if long_term_index < short_term_index:
        data.loc[long_term_index:, 'class'] = 'Long Term'
        data.loc[short_term_index:, 'class'] = 'Short Term'
    else:
        data.loc[short_term_index:, 'class'] = 'Short Term'
        data.loc[long_term_index:, 'class'] = 'Long Term'
    data = data[~data['Date sold or disposed of'].isna()]
    data['month'] = data['Date sold or disposed of'].astype(str).str[0:7]

    # For mapping ERC721, ERC1155 Token
    data['asset'] = data['Description of property'].str[0:-25]
    for token_name in nft_list:
        data.loc[data['asset'].str.contains(token_name), 'asset'] = token_name

    data = pd.merge(data, mapping, how='left', left_on='asset', right_on='cryptoCurrency')
    data = data.groupby(['month', 'asset', 'class', 'quickbookAccount'], as_index=False)['Gain or (loss)'].sum()
    data['account'] = '11000 Cryptocurrency:' + data['asset']
    data['opp_account'] = '80000 Crypto Gains/(Losses):Crypto - Capital Gains & Losses:' \
                          + data['asset'] + ' - Net Capital Gain & Loss'
    data['account'] = data['quickbookAccount']
    long_term_data = data[data['class'] == 'Long Term']
    short_term_data = data[data['class'] == 'Short Term']
    return short_term_data, long_term_data


def generate_monthly_iif(cgl_df):
    iif = pd.DataFrame({'specifier': [], 'date': [], 'tx_type': [],
                        'doc_num': [], 'account': [],
                        'amount': [], 'memo': []})
    end_trans = pd.DataFrame({'specifier': ['ENDTRNS']})
    cgl_df.reset_index(inplace=True, drop=True)
    month_dict = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June',
                  '07': 'July',
                  '08': 'August', '09': 'September', '10': 'October', '11': 'November', '12': 'December'}
    cgl_df['memo_suffix'] = np.nan
    cgl_df.loc[cgl_df['Gain or (loss)'] >= 0, 'memo_suffix'] = 'Net Capital Gain'
    cgl_df.loc[cgl_df['Gain or (loss)'] < 0, 'memo_suffix'] = 'Net Capital LOSS'
    oppo_data = cgl_df.copy()
    oppo_data['Gain or (loss)'] = -oppo_data['Gain or (loss)']
    oppo_data['account'] = oppo_data['opp_account']
    iif = pd.concat([iif, cgl_df, oppo_data], ignore_index=True)
    # iif = pd.concat([iif, cgl_df])
    # print(cgl_df)
    iif['specifier'] = 'SPL'
    iif.loc[0, 'specifier'] = 'TRNS'
    iif['tx_type'] = 'GENERAL JOURNAL'
    iif['day'] = iif['month'].apply(lambda x: calendar.month(int(x[0:4]), int(x[-2:]))[-3:-1])
    iif['date'] = iif['month'].str.replace('-', '/')
    iif['date'] = iif['date'] + '/' + iif['day']
    iif['memo'] = iif['month'].apply(lambda x: month_dict[x[-2:]]) + ' ' + iif['month'].str[0:4] + ' ' \
                  + iif['asset'] + ' ' + iif['memo_suffix']
    iif['amount'] = iif['Gain or (loss)']
    iif['doc_num'] = iif['month'].apply(lambda x: month_dict[x[-2:]][0:3].upper()) + iif['month'].str[0:4] + 'CG&L'
    iif = iif[['specifier', 'date', 'tx_type',
                        'doc_num', 'account',
                        'amount', 'memo']]
    iif = pd.concat([iif, end_trans], ignore_index=True)
    return iif


def convert_cgl_to_iif(file_name, mapping_file):
    short_term, long_term = get_grouped_cgl(file_name, mapping_file)
    if short_term is None:
        return None, None
    iif_header = pd.DataFrame({'specifier': ['!TRNS', '!SPL', '!ENDTRNS'], 'date': ['DATE', 'DATE', ''],
                               'tx_type': ['TRNSTYPE', 'TRNSTYPE', ''],
                               'doc_num': ['DOCNUM', 'DOCNUM', ''], 'account': ['ACCNT', 'ACCNT', ''],
                               'amount': ['AMOUNT', 'AMOUNT', ''], 'memo': ['MEMO', 'MEMO', ''],
                               'name': ['NAME', 'NAME', '']})
    st_month_list = short_term['month'].drop_duplicates().to_list()
    lt_month_list = long_term['month'].drop_duplicates().to_list()
    short_term_iif = iif_header.copy()
    long_term_iif = iif_header.copy()
    for month in st_month_list:
        month_data = short_term[short_term['month'] == month]
        month_iif = generate_monthly_iif(month_data)
        short_term_iif = pd.concat([short_term_iif, month_iif], ignore_index=True)

    for month in lt_month_list:
        month_data = long_term[long_term['month'] == month]
        month_iif = generate_monthly_iif(month_data)
        long_term_iif = pd.concat([long_term_iif, month_iif], ignore_index=True)
    output_file_name = file_name.replace('.csv', '_iif.xlsx')
    with pd.ExcelWriter(output_file_name, mode='w') as writer:
        short_term_iif.to_excel(writer, sheet_name='Short Term IIF', index=False, header=False)
    with pd.ExcelWriter(output_file_name, mode='a', if_sheet_exists='replace') as writer:
        long_term_iif.to_excel(writer, sheet_name='Long Term IIF', index=False, header=False)
    return long_term_iif, short_term_iif


# if __name__ == '__main__':
#     path = '/Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/'
#     pd.set_option('display.max_columns', None)
#     pd.options.mode.chained_assignment = None
#     # file_name = path + '2022 FIFO.xlsx'
#     file_name2 = path + '8949/' + '62e450d070fa9d1da684b2a7_2022_FIFO.csv'
#     mapping = pd.read_excel(path + 'mapping.xlsx', sheet_name='Wallet Directory')
#     mapping = mapping[~mapping['cryptoCurrency'].isna()][['cryptoCurrency', 'quickbookAccount']]
#     nft_list = mapping[(mapping['quickbookAccount'].str.contains('1155'))
#                        | (mapping['quickbookAccount'].str.contains('721'))]['cryptoCurrency'].tolist()
#     convert_cgl_to_iif(file_name2)

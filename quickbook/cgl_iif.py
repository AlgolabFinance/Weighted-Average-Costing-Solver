import pandas as pd
import calendar

def get_grouped_cgl(file_name, sheet_name):
    data = pd.read_excel(file_name, sheet_name=sheet_name)
    data['month'] = data['Date sold or disposed of'].astype(str).str[0:7]
    for i, row in data.iterrows():
        asset = row['Description of property'].split('-')[0]
        data.loc[i, 'asset'] = asset
    data = data.groupby(['month', 'asset'], as_index=False)['Gain or (loss)'].sum()
    data['account'] = '11000 Cryptocurrency:' + data['asset']
    data['opp_account'] = '80000 Crypto Gains/(Losses):Crypto - Capital Gains & Losses:' \
                          + data['asset'] + ' - Net Capital Gain & Loss'
    return data

def generate_monthly_iif(cgl_df):
    iif = pd.DataFrame({'specifier': [], 'date': [], 'tx_type': [],
                        'doc_num': [], 'account': [],
                        'amount': [], 'memo': []})
    end_trans = pd.DataFrame({'specifier': ['ENDTRANS']})
    cgl_df.reset_index(inplace=True, drop=True)
    month_dict = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June',
                  '07': 'July',
                  '08': 'August', '09': 'September', '10': 'October', '11': 'November', '12': 'December'}
    for i, row in cgl_df.iterrows():
        specifier = None
        if i == 0:
            specifier = 'TRANS'
        else:
            specifier = 'SPL'
        print(row)
        day = calendar.month(int(row['month'][0:4]), int(row['month'][-2:]))[-3:-1]
        date = row['month'].replace('-', '/') + '/' + day
        account = row['account']
        opp_account = row['opp_account']
        amount = row['Gain or (loss)']
        str_month = row['month'][-2:]
        str_year = row['month'][0:4]
        memo = month_dict[str_month] + ' ' + str_year + ' ' + row['asset'] + ' Net Capital Gain'
        tx_type = 'GENERAL JOURNAL'
        doc_num = month_dict[str_month][0:3].upper() + str_year + 'CG&L'
        new_row = pd.DataFrame({'specifier': [specifier], 'date': [date], 'tx_type': [tx_type], 'doc_num': [doc_num],
                                'account': [account], 'amount': [amount], 'memo': [memo]})
        specifier = 'SPL'
        new_opp_row = pd.DataFrame(
            {'specifier': [specifier], 'date': [date], 'tx_type': [tx_type], 'doc_num': [doc_num],
             'account': [opp_account], 'amount': [-amount], 'memo': [memo]})
        iif = pd.concat([iif, new_row, new_opp_row], ignore_index=True)
    iif = pd.concat([iif, end_trans], ignore_index=True)
    return iif


def convert_cgl_to_iif(file_name):
    short_term = get_grouped_cgl(file_name, 'Short Term')
    long_term = get_grouped_cgl(file_name, 'Long Term')
    iif_header = pd.DataFrame({'specifier': ['!TRNS', '!SPL', '!ENDTRANS'], 'date': ['DATE', 'DATE', ''],
                               'tx_type': ['TRNSTYPE', 'TRNSTYPE', ''],
                               'doc_num': ['DOCNUM', 'DOCNUM', ''], 'account': ['ACCNT', 'ACCNT', ''],
                               'amount': ['AMOUNT', 'AMOUNT', ''], 'memo': ['MEMO', 'MEMO', '']})
    st_month_list = short_term['month'].drop_duplicates().to_list()
    lt_month_list = long_term['month'].drop_duplicates().to_list()
    short_term_iif = iif_header.copy()
    long_term_iif = iif_header.copy()
    #print(short_term)
    for month in st_month_list:
        month_data = short_term[short_term['month'] == month]
        month_iif = generate_monthly_iif(month_data)
        short_term_iif = pd.concat([short_term_iif, month_iif], ignore_index=True)

    for month in lt_month_list:
        month_data = long_term[long_term['month'] == month]
        month_iif = generate_monthly_iif(month_data)
        long_term_iif = pd.concat([long_term_iif, month_iif], ignore_index=True)

    with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='replace') as writer:
        short_term_iif.to_excel(writer, sheet_name='Short Term IIF', index=False, header=False)
        long_term_iif.to_excel(writer, sheet_name='Long Term IIF', index=False, header=False)
    return long_term_iif, short_term_iif


if __name__ == '__main__':
    path = '/Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/'
    pd.set_option('display.max_columns', None)
    file_name = path + '2022 FIFO.xlsx'
    long_term_iif, short_term_iif = convert_cgl_to_iif(file_name)
    long_term_iif.to_csv('long_term_iif.iif', sep='\t')
    short_term_iif.to_csv('short_term_iif.iif', sep='\t')

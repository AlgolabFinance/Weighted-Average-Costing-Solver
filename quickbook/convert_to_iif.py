import pandas as pd
import calendar
from dateutil.parser import parse
from cgl_iif import *



def convert_to_iif(file_name, is_debit_only=False):
    data = None
    try:
        if is_debit_only:
            debit_data = pd.read_excel(file_name, sheet_name='Debit', dtype={'date': 'str'})
            credit_data = None
        else:
            debit_data = pd.read_excel(file_name, sheet_name='Debit', dtype={'date': 'str'})
            credit_data = pd.read_excel(file_name, sheet_name='Credit', dtype={'date': 'str'})
    except Exception:
        print('Empty File Or Wrong Sheet Name: ' + file_name)
        return None, None

    iif = pd.DataFrame({'specifier':['!TRNS', '!SPL', '!ENDTRNS'], 'date': ['DATE', 'DATE', ''], 'tx_type':['TRNSTYPE','TRNSTYPE', ''],
                         'doc_num': ['DOCNUM', 'DOCNUM', ''],
                        'account': ['ACCNT', 'ACCNT', ''],
                'amount': ['AMOUNT', 'AMOUNT', ''], 'memo': ['MEMO', 'MEMO', ''], 'name': ['NAME', 'NAME', '']})
    #trans = pd.DataFrame({'specifier':['TRNS'], 'date':[''], 'account':[''], 'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})

    end_trans = pd.DataFrame({'specifier':['ENDTRNS'], 'date':[''], 'account':[''],
                              'tx_type':[''],'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})
    debit_iif, credit_iif = iif.copy(), iif.copy()
    if debit_data is not None:
        for i, row in debit_data.iterrows():
            specifier = 'TRNS'
            account = row['Account']
            date = parse(row['date'])
            tx_type = 'Deposit'
            year = date.year
            month = date.month
            day = calendar.month(int(year), int(month))[-3:-1]
            date = str(month) + '/' + str(day) + '/' + str(year)
            amount = round(row['Amount(USD)'], 2)
            memo = row['memo']
            new_trans_row = pd.DataFrame({'specifier':[specifier], 'date':[date], 'tx_type':[tx_type],
                                    'account':[account],  'amount':[amount], 'memo':[memo]})
            specifier = 'SPL'
            account = row['oppAccount']
            new_spl_row = pd.DataFrame({'specifier':[specifier], 'date':[date],'tx_type':[tx_type],
                                    'account':[account],  'amount':[-amount], 'memo': [memo]})
            debit_iif = pd.concat([debit_iif, new_trans_row, new_spl_row, end_trans], ignore_index=True)

    if credit_data is not None:
        for i, row in credit_data.iterrows():
            specifier = 'TRNS'
            account = row['Account']
            date = parse(row['date'])
            tx_type = 'Check'
            year = date.year
            month = date.month
            day = calendar.month(int(year), int(month))[-3:-1]
            date = str(month) + '/' + str(day) + '/' + str(year)
            amount = row['Amount(USD)']
            memo = row['memo']
            new_trans_row = pd.DataFrame({'specifier':[specifier], 'date': [date], 'tx_type':[tx_type],
                                    'account':[account],  'amount': [-amount], 'memo':[memo]})
            specifier = 'SPL'
            account = row['oppAccount']
            new_spl_row = pd.DataFrame({'specifier':[specifier], 'date':[date], 'tx_type':[tx_type],
                                    'account':[account],  'amount':[amount], 'memo':[memo]})
            credit_iif = pd.concat([credit_iif, new_trans_row, new_spl_row, end_trans], ignore_index=True)

    with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='replace') as writer:
        debit_iif.to_excel(writer, sheet_name='DEBIT IIF', index=False, header=False)
        credit_iif.to_excel(writer, sheet_name='CREDIT IIF', index=False, header=False)

    return debit_iif, credit_iif


def is_debit_only(file_name):
    keywords = ['trade', 'buy', 'sell', 'convert', 'transfer']
    file_name = file_name.lower()
    for keyword in keywords:
        if keyword in file_name:
            return True
    return False


def convert_all_files(path, is_cgl=False):
    file_list = glob.glob(path + '*.xlsx')
    if is_cgl:
        for file_name in file_list:
            prefix = file_name.split('.')[-2]
            long_term_iif, short_term_iif = convert_cgl_to_iif(file_name)
            # long_term_iif['amount'] = long_term_iif['amount'].apply(lambda x: round(x, 2))
            # short_term_iif['amount'] = long_term_iif['amount'].apply(lambda x: round(x, 2))
            if long_term_iif is not None:
                long_term_iif.to_csv(prefix + '_long_term.iif', sep='\t', index=False, header=False)
            if short_term_iif is not None:
                short_term_iif.to_csv(prefix + '_short_term.iif', sep='\t', index=False, header=False)
    else:
        for file_name in file_list:
            prefix = file_name.split('.')[-2]
            if is_debit_only(file_name):
                debit_iif, _ = convert_to_iif(file_name, True)
#                debit_iif['amount'] = debit_iif['amount'].apply(lambda x: round(x, 2))
                if debit_iif is not None:
                    debit_iif.to_csv(prefix + '.iif')
            else:
                debit_iif, credit_iif = convert_to_iif(file_name, False)
                # debit_iif['amount'] = debit_iif['amount'].apply(lambda x: round(x, 2))
                # credit_iif['amount'] = credit_iif['amount'].apply(lambda x: round(x, 2))
                if debit_iif is not None:
                    debit_iif.to_csv(prefix + '_debit.iif')
                if credit_iif is not None:
                    credit_iif.to_csv(prefix + '_credit_iif')



if __name__ == '__main__':
    path = '/Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/'
    pd.set_option('display.max_columns', None)
    # Set is_cgl as False if it is a quick book directory
    convert_all_files(path + 'QB/', False)

    # Set is_cgl as True if it is an 8949 directory
    convert_all_files(path + '8949/', True)

    # file0 = path + '202110_Deposit_Withdrawal.xlsx'
    # res0_debit, res0_credit = convert_to_iif(file0)
    # res0_debit.to_csv(path + '202110_Deposit_Withdrawal_Debit.iif', sep='\t', index=False, header=False)
    # res0_credit.to_csv(path + '202110_Deposit_Withdrawal_Credit.iif', sep='\t', index=False, header=False)
    # cal = calendar.month(2021, 10)
    # #print(cal)
    # file1 = path + '202102_Deposit_Withdrawal.xlsx'
    # file2 = path + '202102_Trade_Buy_Sell_Convert.xlsx'
    # file3 = path + '202102_Transfer.xlsx'
    # res1_debit, res1_credit = convert_to_iif(file1)
    # res1_debit.to_csv(path + 'debit_deposit_withdrawal.iif', sep='\t')
    # res1_credit.to_csv(path + 'credit_deposit_withdrawal.iif', sep='\t')
    # res2_debit, res2_credit = convert_to_iif(file2)
    # convert_to_iif(file3, True)
    # res3 = convert_to_iif(file3, True)
    # res1.to_csv(path + 'deposit_withdrawal.iif', sep='\t')
    # res2.to_csv(path + 'trade_buy_sell_convert.iif', sep='\t')
    # res3.to_csv(path + 'transfer.iif', sep='\t')




import pandas as pd
import calendar
from dateutil.parser import parse


# structure_dict = {'specifier':[], 'date':[], 'account':[], 'docnum':[], 'class':[], 'amount':[], 'amount_memo':[]}
# iif = pd.DataFrame(structure_dict)
def convert_to_iif(file_name, is_transfer=False):
    data = None
    if is_transfer:
        debit_data = pd.read_excel(file_name, sheet_name='Debit', dtype=str)
        credit_data = None
    else:
        debit_data = pd.read_excel(file_name, sheet_name='Debit', dtype=str)
        credit_data = pd.read_excel(file_name, sheet_name='Credit', dtype=str)
    iif = pd.DataFrame({'specifier':['!TRNS', '!SPL', '!ENDTRANS'], 'date': ['DATE', 'DATE', ''], 'tx_type':['TRNSTYPE','TRNSTYPE', ''],
                        'account': ['ACCNT', 'ACCNT', ''],
                'amount': ['AMOUNT', 'AMOUNT', ''], 'memo': ['MEMO', 'MEMO', '']})
    #trans = pd.DataFrame({'specifier':['TRANS'], 'date':[''], 'account':[''], 'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})

    end_trans = pd.DataFrame({'specifier':['ENDTRANS'], 'date':[''], 'account':[''],
                              'tx_type':[''],'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})
    debit_iif, credit_iif = iif.copy(), iif.copy()
    if debit_data is not None:
        for i, row in debit_data.iterrows():
            specifier = 'TRANS'
            account = row['Account']
            date = parse(row['date'])
            tx_type = 'Deposit'
            year = date.year
            month = date.month
            day = calendar.month(int(year), int(month))[-3:-1]
            date = str(month) + '/' + str(day) + '/' + str(year)
            amount = row['Amount(USD)']
            memo = row['memo']
            new_trans_row = pd.DataFrame({'specifier':[specifier], 'date':[date], 'tx_type':[tx_type],
                                    'account':[account],  'amount':[amount], 'memo':[memo]})
            specifier = 'SPL'
            account = row['oppAccount']
            new_spl_row = pd.DataFrame({'specifier':[specifier], 'date':[date],'tx_type':[tx_type],
                                    'account':[account],  'amount':['-'+amount], 'memo': [memo]})
            debit_iif = pd.concat([debit_iif, new_trans_row, new_spl_row, end_trans], ignore_index=True)

    if credit_data is not None:
        for i, row in credit_data.iterrows():
            specifier = 'TRANS'
            account = row['Account']
            date = parse(row['date'])
            tx_type = 'Check'
            year = date.year
            month = date.month
            day = calendar.month(int(year), int(month))[-3:-1]
            date = str(month) + '/' + str(day) + '/' + str(year)
            amount = row['Amount(USD)']
            memo = row['memo']
            new_trans_row = pd.DataFrame({'specifier':[specifier], 'date':[date], 'tx_type':[tx_type],
                                    'account':[account],  'amount':['-' + amount], 'memo':[memo]})
            specifier = 'SPL'
            account = row['oppAccount']
            new_spl_row = pd.DataFrame({'specifier':[specifier], 'date':[date], 'tx_type':[tx_type],
                                    'account':[account],  'amount':[amount], 'memo':[memo]})
            credit_iif = pd.concat([credit_iif, new_trans_row, new_spl_row, end_trans], ignore_index=True)

    with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='replace') as writer:
        debit_iif.to_excel(writer, sheet_name='DEBIT IIF', index=False, header=False)
        credit_iif.to_excel(writer, sheet_name='CREDIT IIF', index=False, header=False)

    print(credit_iif['memo'])
    return debit_iif, credit_iif

if __name__ == '__main__':
    path = '/Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/'
    pd.set_option('display.max_columns', None)
    file0 = path + '202110_Deposit_Withdrawal.xlsx'
    res0_debit, res0_credit = convert_to_iif(file0)
    res0_debit.to_csv(path + '202110_Deposit_Withdrawal_Debit.iif', sep='\t', index=False, header=False)
    res0_credit.to_csv(path + '202110_Deposit_Withdrawal_Credit.iif', sep='\t', index=False, header=False)
    cal = calendar.month(2021, 10)
    #print(cal)
    # file1 = path + '202102_Deposit_Withdrawal.xlsx'
    # file2 = path + '202102_Trade_Buy_Sell_Convert.xlsx'
    # file3 = path + '202102_Transfer.xlsx'
    # res1 = convert_to_iif(file1)
    # res2 = convert_to_iif(file2)
    # res3 = convert_to_iif(file3, True)
    # res1.to_csv(path + 'deposit_withdrawal.iif', sep='\t')
    # res2.to_csv(path + 'trade_buy_sell_convert.iif', sep='\t')
    # res3.to_csv(path + 'transfer.iif', sep='\t')




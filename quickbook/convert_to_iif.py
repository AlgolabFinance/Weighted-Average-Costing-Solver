import pandas as pd



# structure_dict = {'specifier':[], 'date':[], 'account':[], 'docnum':[], 'class':[], 'amount':[], 'amount_memo':[]}
# iif = pd.DataFrame(structure_dict)
def convert_to_iif(file_name, is_transfer=False):
    data = None
    if is_transfer:
        data = pd.read_excel(file_name, sheet_name='Debit', dtype=str)
    else:
        debit = pd.read_excel(file_name, sheet_name='Debit')
        credit = pd.read_excel(file_name, sheet_name='Debit')
        data = pd.concat([debit, credit], ignore_index=True)
    print(data)
    iif = pd.DataFrame({'specifier':['!TRNS', '!SPL','!ENDTRANS'], 'date':['DATE', 'DATE', ''], 'account':['ACCNT', 'ACCNT', ''],
               'docnum':['DOCNUM', 'DOCNUM', ''], 'class':['CLASS', 'CLASS', ''], 'amount':['AMOUNT', 'AMOUNT', ''], 'amount_memo':['AMOUNT MEMO', 'AMOUNT MEMO', '']})
    trans = pd.DataFrame({'specifier':['TRANS'], 'date':[''], 'account':[''], 'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})

    end_trans = pd.DataFrame({'specifier':['ENDTRANS'], 'date':[''], 'account':[''], 'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})
    for i, row in data.iterrows():
        specifier = 'TRANS'
        account = row['Account']
        date = row['date']
        amount = row['Amount(USD)']
        memo = row['memo']
        docnum = ''
        class_type = 'GENERAL JOURNAL'
        new_row = pd.DataFrame({'specifier':[specifier], 'date':[date],
                                'account':[account], 'docnum':[docnum], 'class':[class_type], 'amount':[amount], 'amount_memo':[memo]})
        iif = pd.concat([iif, new_row, end_trans], ignore_index=True)
    return iif

if __name__ == '__main__':
    path = '/Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/'
    pd.set_option('display.max_columns', None)
    file1 = path + '202102_Deposit_Withdrawal.xlsx'
    file2 = path + '202102_Trade_Buy_Sell_Convert.xlsx'
    file3 = path + '202102_Transfer.xlsx'
    res1 = convert_to_iif(file1)
    res2 = convert_to_iif(file2)
    res3 = convert_to_iif(file3, True)
    res1.to_csv(path + 'deposit_withdrawal.iif', sep='\t')
    res2.to_csv(path + 'trade_buy_sell_convert.iif', sep='\t')
    res3.to_csv(path + 'transfer.iif', sep='\t')




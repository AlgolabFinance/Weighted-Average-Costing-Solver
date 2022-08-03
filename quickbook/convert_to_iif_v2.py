import pandas as pd
import calendar
from dateutil.parser import parse
from cgl_iif_v2 import *



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

    iif = pd.DataFrame({'specifier':['!TRNS', '!SPL', '!ENDTRANS'], 'date': ['DATE', 'DATE', ''], 'tx_type':['TRNSTYPE','TRNSTYPE', ''],
                         'doc_num': ['DOCNUM', 'DOCNUM', ''],
                        'account': ['ACCNT', 'ACCNT', ''],
                'amount': ['AMOUNT', 'AMOUNT', ''], 'memo': ['MEMO', 'MEMO', ''], 'name': ['NAME', 'NAME', '']})
    #trans = pd.DataFrame({'specifier':['TRANS'], 'date':[''], 'account':[''], 'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})

    end_trans = pd.DataFrame({'specifier':['ENDTRANS'], 'date':[''], 'account':[''],
                              'tx_type':[''],'docnum':[''], 'class':[''], 'amount':[''], 'amount_memo':['']})
    debit_iif, credit_iif = iif.copy(), iif.copy()

    def format_date(str_date):
        date = parse(str_date)
        year = date.year
        month = date.month
        day = calendar.month(int(year), int(month))[-3:-1]
        date = str(month) + '/' + str(day) + '/' + str(year)
        return date

    if debit_data is not None:
        debit_data['specifier'] = 'TRNS'
        debit_data['account'] = debit_data['Account']
        debit_data['tx_type'] = 'Deposit'
        debit_data['date'] = debit_data['date'].apply(format_date)
        debit_data['amount'] = debit_data['Amount(USD)']
        opp_data = debit_data.copy()
        opp_data['account'] = opp_data['oppAccount']
        opp_data['amount'] = -opp_data['amount']
        opp_data['specifier'] = 'SPL'
        opp_data.index = opp_data.index + 0.5
        end = debit_data['specifier'].to_frame()
        end['specifier'] = 'ENDTRNS'
        end.index = end.index + 0.8
        debit_data = pd.concat([debit_data, opp_data, end])
        debit_data.sort_index(inplace=True)
        debit_data.reset_index(inplace=True)
        debit_data['doc_num'] = ''
        debit_data['name'] = ''
        debit_data = debit_data[['specifier', 'date', 'tx_type', 'doc_num', 'account', 'amount', 'memo', 'name']]
        debit_iif = pd.concat([debit_iif, debit_data])

    if credit_data is not None:
        credit_data['specifier'] = 'TRNS'
        credit_data['account'] = credit_data['Account']
        credit_data['tx_type'] = 'Check'
        credit_data['date'] = credit_data['date'].apply(format_date)
        credit_data['amount'] = -credit_data['Amount(USD)']
        opp_data = credit_data.copy()
        opp_data['account'] = opp_data['oppAccount']
        opp_data['amount'] = -opp_data['amount']
        opp_data['specifier'] = 'SPL'
        opp_data.index = opp_data.index + 0.5
        end = credit_data['specifier'].to_frame()
        end['specifier'] = 'ENDTRNS'
        end.index = end.index + 0.8
        credit_data = pd.concat([credit_data, opp_data, end])
        credit_data.sort_index(inplace=True)
        credit_data.reset_index(inplace=True)
        credit_data['doc_num'] = ''
        credit_data['name'] = ''
        credit_data = credit_data[['specifier', 'date', 'tx_type', 'doc_num', 'account', 'amount', 'memo', 'name']]
        credit_iif = pd.concat([credit_iif, credit_data])

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


def convert_all_files(path, is_cgl=False, mapping_file=None):
    if not is_cgl:
        file_list = glob.glob(path + '*.xlsx')
    else:
        file_list = glob.glob(path + '*.csv')
        if mapping_file is None:
            print('Error: Missing the mapping file')
            return
    if is_cgl:
        for file_name in file_list:
            prefix = file_name.split('.')[-2]
            long_term_iif, short_term_iif = convert_cgl_to_iif(file_name, mapping_file)
            if long_term_iif is not None:
                long_term_iif.to_csv(prefix + '_long_term.iif', sep='\t', index=False, header=False)
            if short_term_iif is not None:
                short_term_iif.to_csv(prefix + '_short_term.iif', sep='\t', index=False, header=False)
    else:
        for file_name in file_list:
            prefix = file_name.split('.')[-2]
            if is_debit_only(file_name):
                debit_iif, _ = convert_to_iif(file_name, True)
                if debit_iif is not None:
                    debit_iif.to_csv(prefix + '.iif')
            else:
                debit_iif, credit_iif = convert_to_iif(file_name, False)
                if debit_iif is not None:
                    debit_iif.to_csv(prefix + '_debit.iif')
                if credit_iif is not None:
                    credit_iif.to_csv(prefix + '_credit_iif')


if __name__ == '__main__':
    path = '/Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/'
    pd.set_option('display.max_columns', None)
    pd.options.mode.chained_assignment = None
    # Set is_cgl as False if it is a quick book directory
    convert_all_files(path + 'QB/', False)

    # Set is_cgl as True if it is an 8949 directory
    # Must specify the mapping file
    convert_all_files(path + '8949/', True, path + 'mapping.xlsx')




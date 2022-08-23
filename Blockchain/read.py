# import dependencies
import pickle
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware


# instantiate a web3 remote provider
w3 = Web3(Web3.WebsocketProvider('ws://127.0.0.1:8546'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)
print(w3.isConnected())
# request the latest block number
ending_blocknumber = w3.eth.blockNumber

starting_blocknumber = 0

# filter through blocks and look for transactions involving this address
blockchain_address = "0x7D7eFbD5575d68c146610b3916b793a0325e0DF6"

# create an empty dictionary we will add transaction data to
tx_dictionary = {}

balance = w3.eth.get_balance(blockchain_address, block_identifier='latest')
print(balance)
block = w3.eth.get_block('latest')
# print(block)


def getTransactions(start, end, address):
    '''This function takes three inputs, a starting block number, ending block number
    and an Ethereum address. The function loops over the transactions in each block and
    checks if the address in the to field matches the one we set in the blockchain_address.
    Additionally, it will write the found transactions to a pickle file for quickly serializing and de-serializing
    a Python object.'''
    print(
        f"Started filtering through block number {start} to {end} for transactions involving the address - {address}...")
    for x in range(start, end):
        block = w3.eth.get_block(x, full_transactions=True)
        for transaction in block['transactions']:
            # print(transaction)
            if transaction['to'] == address or transaction['from'] == address:
                print(transaction)
                with open("transactions-5.pkl", "wb") as f:
                    hashStr = transaction['hash'].hex()
                    tx_dictionary[hashStr] = transaction
                    pickle.dump(tx_dictionary, f)
                    print(x)
                f.close()
    print(f"Finished searching blocks {start} through {end} and found {len(tx_dictionary)} transactions")


getTransactions(starting_blocknumber, ending_blocknumber, blockchain_address)
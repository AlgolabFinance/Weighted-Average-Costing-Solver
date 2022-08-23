import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
dataset = pd.read_csv("full_dataset.csv")
dataset = dataset.drop(columns=['internal_count', 'normal_count', 'erc20_count', 'erc721_count', 'method'])
dataset = dataset[dataset['rule'].str.contains('\?') == False]
dataset['rule'] = dataset['rule'].str.replace('0X', '0x')
# dataset.loc[dataset['method'].str.len() > 50, 'method'] = 'unknown'
# dataset['method'] = dataset['method'].fillna('empty')
dataset['method_id'] = dataset['method_id'].fillna('empty')
dataset = dataset[dataset['method_id'] != 'deprecated']

# dataset['rule'] = dataset['rule'].str.replace('MINTNFT', 'WITHDRAW')

# drop none value for rule
dataset.dropna(subset=['rule'], inplace=True)
dataset['rule'] = dataset['rule'].str.upper()
dataset['rule'] = dataset['rule'].str.strip()
dataset['rule'] = dataset['rule'].str.replace('MINTNFT', 'MINT NFT')
dataset['rule'] = dataset['rule'].str.replace('WITHDRAWAL', 'WITHDRAW')
dataset['rule'] = dataset['rule'].str.replace('NFTMINTING', 'NFT MINTING')
dataset['rule'] = dataset['rule'].str.replace('WRAPMATIC', 'WRAP MATIC')
dataset['rule'] = dataset['rule'].str.replace('OTHERS', 'OTHER')
rules = dataset['rule'].drop_duplicates()
print(len(rules))
print(len(dataset))
for rule in rules:
    dataset.loc[dataset['rule'] == rule, 'count'] = len(dataset[dataset['rule'] == rule])

# Drop insufficient category
dataset = dataset[dataset['count'] > 9]

print(len(dataset.drop_duplicates(subset=['rule'])[['rule', 'count']]))
dataset.drop(columns='count', inplace=True)
X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]
print(len(X['method_id'].drop_duplicates()))
X = pd.get_dummies(X, drop_first=True)
# print(X.columns)
print(len(X))
# print(y)
# print(X.head(1))
# Split training set, test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5, stratify=y)

# Create classifier
clf = DecisionTreeClassifier(criterion='entropy')

# Train model
clf.fit(x_train, y_train)

# Prediction
y_pred = clf.predict(x_test)

# Evaluate Model
accuracy = np.round(accuracy_score(y_test, y_pred), 3)
print("Accuracy of the new classifier =", accuracy)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).T.to_excel('report_new.xlsx')

predictions = clf.predict_proba(x_test)


# 'ovr':
# Stands for One-vs-rest. Computes the AUC of each class against the rest [3] [4].
# This treats the multiclass case in the same way as the multilabel case. Sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
#
# 'ovo':
# Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes [5].
# Insensitive to class imbalance when average == 'macro'.
auc_score = roc_auc_score(y_test, predictions, multi_class='ovr')

print('AUC Score of DTree: ' + str(auc_score))

rfc = RandomForestClassifier(criterion='entropy', random_state=0, n_estimators=100)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)
accuracy = np.round(accuracy_score(y_test, y_pred_rfc), 3)
report_rfc = classification_report(y_test, y_pred_rfc, output_dict=True)
pd.DataFrame(report_rfc).T.to_excel('report_rfc.xlsx')
print('Accuracy of RFC: ' + str(accuracy))

predictions = rfc.predict_proba(x_test)
auc_score = roc_auc_score(y_test, predictions, multi_class='ovr')

print('auc_score of RFC: ' + str(auc_score))

k = 5
acc_scores = []
auc_scores = []
kf = KFold(n_splits=k, shuffle=True, random_state=3)
X.reset_index(inplace=True, drop=True)
y = y.to_frame().reset_index(drop=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.loc[train_index, 'rule'], y.loc[test_index, 'rule']
    # print(X_train.head(1))
    clf.fit(X_train, y_train)
    pred_values = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    acc = accuracy_score(pred_values, y_test)
    auc = roc_auc_score(y_test, pred_prob, multi_class='ovr')
    auc_scores.append(auc)
    acc_scores.append(acc)

avg_acc_score = sum(acc_scores)/k
avg_auc_score = sum(auc_scores)/k

print('Avg accuracy of decision tree(KFold): {}'.format(avg_acc_score))
print('Avg AUC score of decision tree(KFold): {}'.format(avg_auc_score))

acc_scores = []
auc_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.loc[train_index, 'rule'], y.loc[test_index, 'rule']
    rfc.fit(X_train, y_train)
    pred_values = rfc.predict(X_test)
    pred_prob = rfc.predict_proba(X_test)
    acc = accuracy_score(pred_values, y_test)
    auc = roc_auc_score(y_test, pred_prob, multi_class='ovr')
    auc_scores.append(auc)
    acc_scores.append(acc)

avg_acc_score = sum(acc_scores)/k
avg_auc_score = sum(auc_scores)/k

print('Avg accuracy of random forest(KFold): {}'.format(avg_acc_score))
print('Avg AUC score of random forest(KFold): {}'.format(avg_auc_score))
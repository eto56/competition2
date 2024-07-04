
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
 
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import gc

import warnings
warnings.filterwarnings("ignore")

INPUT_DIR = "./data/"

train = pd.read_csv(INPUT_DIR + "train.csv")
test = pd.read_csv(INPUT_DIR + "test.csv")
sample_sub = pd.read_csv(INPUT_DIR + "sample_submission.csv")

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample submission shape:", sample_sub.shape)
 

"""## Exploratory Data Analysis"""

train.head()

(train['TARGET'].value_counts() / len(train)).to_frame()

sns.countplot(train, x="TARGET", palette='Accent')
plt.show()

train.shape

numeric_features = train.select_dtypes(include='number').columns
train[numeric_features].describe()

train['CODE_GENDER'].value_counts()

train['CODE_GENDER'] = train['CODE_GENDER'].replace('XNA', 'F')
test['CODE_GENDER'] = test['CODE_GENDER'].replace('XNA', 'F')
train['CODE_GENDER'].value_counts()

train.boxplot(column=['AMT_INCOME_TOTAL'])
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, ',')))
plt.show()

pd.set_option('display.float_format', lambda x: '%.2f' % x)
train['AMT_INCOME_TOTAL'].describe()

train['AMT_INCOME_TOTAL'].sort_values(ascending=False).head()
print(0)
print(train.shape)
print(test.shape)

train = train[train['AMT_INCOME_TOTAL'] < 5_000_000]
#test = test[test['AMT_INCOME_TOTAL'] < 5_000_000]


#欠損値は中央値で補完
def handle_missing_values(df):
    numeric_features = df.select_dtypes(include='number').columns
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

    categorical_features = df.select_dtypes(include=['object']).columns
    df[categorical_features] = df[categorical_features].fillna('Unknown')

    return df

train = handle_missing_values(train)
test = handle_missing_values(test)

train.dtypes.value_counts()

numeric_features = train.select_dtypes(include='number').columns.tolist()
categorical_features = train.select_dtypes(include='object').columns.tolist()

print('Count of numeric features:', len(numeric_features))
print('Count of categorical features:', len(categorical_features))
print('Total:', len(numeric_features) + len(categorical_features))

categorical_features

train.groupby('CODE_GENDER', as_index=False)['TARGET'].value_counts()

train.groupby('CODE_GENDER', as_index=False)['TARGET'].value_counts().pivot(
    index='CODE_GENDER',
    columns='TARGET',
    values='count'
).plot(
    kind='barh',
)

train.groupby('FLAG_OWN_CAR', as_index=False)['TARGET'].value_counts()

train.groupby('FLAG_OWN_CAR', as_index=False)['TARGET'].value_counts().pivot(
    index='FLAG_OWN_CAR',
    columns='TARGET',
    values='count'
).plot(
    kind='barh',
)

train.groupby('FLAG_OWN_REALTY', as_index=False)['TARGET'].value_counts()

train.groupby('FLAG_OWN_REALTY', as_index=False)['TARGET'].value_counts().pivot(
    index='FLAG_OWN_REALTY',
    columns='TARGET',
    values='count'
).plot(
    kind='barh',
)

 
"""## One-hot encoding categorical features"""

one_hot_encoded = pd.concat([train,test])
one_hot_encoded = pd.get_dummies(
    one_hot_encoded,
    columns=categorical_features,
    drop_first=True,
    sparse=True
)

one_hot_encoded.head(10).transpose()

train.dtypes.value_counts()
 

train = one_hot_encoded[:train.shape[0]]
test = one_hot_encoded[train.shape[0]:]
print(1)
print(train.shape)
print(test.shape)
one_hot_encoded = None
gc.collect()

y_train = train['TARGET'].values
train = train.drop(columns=['TARGET'])
x_test = test.copy()
print(2)
print(train.shape)
print(x_test.shape)

"""## LightGBM Model"""

 
x_train, x_valid, y_train, y_valid = train_test_split(train, y_train, test_size=0.8, stratify=y_train, random_state=1)
print('Train shape:', x_train.shape)
print('Test shape:', x_train.shape)
print('Valid shape:', x_valid.shape)

 
x_train = x_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
x_test = x_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

clf = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
    )

clf.fit(x_train,y_train, eval_metric='auc')

"""## Variable importances"""

importances = clf.feature_importances_

sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = x_train.columns[sorted_indices]

top_features = sorted_features[:10]
top_importances = sorted_importances[:10]

 
print(top_features)
print(top_importances)

 

"""## Submission"""

y_predict = clf.predict_proba(x_test[x_train.columns])[:, 1]
print(y_predict.__len__())
lgb_submission = pd.DataFrame({'SK_ID_CURR': x_test['SK_ID_CURR'].astype(int), 'TARGET': y_predict})
lgb_submission.head()
print(lgb_submission.shape)
lgb_submission.to_csv('lgb_submission.csv', index=False)


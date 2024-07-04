
# ライブラリの読み込み
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
 
INPUT_DIR = "./data/"

train = pd.read_csv(INPUT_DIR + "train.csv")
test = pd.read_csv(INPUT_DIR + "test.csv")
sample_sub = pd.read_csv(INPUT_DIR + "sample_submission.csv")

  
use_features = train.select_dtypes(include='number').columns

use_features = use_features.drop('SK_ID_CURR')
train = train[use_features]
print("start")
for col in train.columns:
    print(col)
    sns.countplot(data=train, x=col)
    #to_png
    print(f'./data/figures/{col}.png')
    plt.savefig(f'./data/figures/{col}.png')
    print("end")
    
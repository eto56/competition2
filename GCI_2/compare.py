import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
 

import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


take2=pd.read_csv("./take2.csv")
take3=pd.read_csv("./take3.csv")
take4=pd.read_csv("./take4.csv")

all=take2


all["TARGET_2"]=take2["TARGET"]
all["TARGET_3"]=take3["TARGET"]
all["TARGET_4"]=take4["TARGET"]
#all.drop("SK_ID_CURR",axis=1,inplace=True)
all.drop("TARGET",axis=1,inplace=True)
 
ind=all["SK_ID_CURR"].values
t2=all["TARGET_2"].values
t3=all["TARGET_3"].values
t4=all["TARGET_4"].values

#折れ線グラフ
#figsize
plt.figure(figsize=(20,10))
plt.plot(ind,t2,label="take2")
plt.plot(ind,t3,label="take3")
plt.plot(ind,t4,label="take4")
plt.legend()
plt.savefig("result.png")
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('fivethirtyeight')
import warnings
#warnings.filterwarnings('ignore')
#%matplotlib inline

import xgboost as xg


def load_data():
    data = pd.read_csv('data/train.csv')

    d_t = data.head()

    print(d_t)

    return d_t


load_data()

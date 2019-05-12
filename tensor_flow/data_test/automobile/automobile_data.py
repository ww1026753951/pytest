import os
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import train_test_split

from io import BytesIO

import numpy as np

import seaborn as sb


import collections

sb.set_context("notebook", font_scale=2.5)

from matplotlib import pyplot as plt
from natsort import natsorted


data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)
])

def raw_dataframe():
  """Load the automobile data set as a pd.DataFrame."""
  # Download and cache the data
  # # 获取数据路径
  # path = tf.keras.utils.get_file(fname=os.path.basename(data_url), origin=data_url)


  df = pd.read_csv(data_url, names=COLUMN_TYPES.keys(), sep=',', na_values = "?")
  #


  return df


def show_data(df):
    # 删除空数据
    df = df.dropna()

    # df["price"].fillna(0)
    df.info()

    # 打印数据总体概览
    print(df.describe().T)

    df.hist(bins=50, figsize=(20, 15))
    # plt.show()
    # column_list = list(df.columns)[0:-1]
    column_list = list(df.columns)

    sb.pairplot(df.loc[:, column_list], height=5)

    # plt.show()
    plt.savefig('d:/a.png')

# 读取数据,通过pandas 的 sample  api切分数据
def load_data_split_by_sample(y_name="price", train_fraction=0.7, seed=None):

  # Load the raw data columns.
  data = raw_dataframe()

  # 数据可视化
  show_data(data)

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  # np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features DataFrame.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)




def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


if __name__ == '__main__':
    load_data_split_by_sample()

import os
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import train_test_split

from io import BytesIO

import numpy as np

import seaborn as sb



sb.set_context("notebook", font_scale=2.5)

from matplotlib import pyplot as plt
from natsort import natsorted


data_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

# data_url = "d:/doc/winequality-red.csv"

CSV_COLUMN_NAMES = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income',
                   'median_house_value', 'ocean_proximity']


def raw_dataframe():
  """Load the automobile data set as a pd.DataFrame."""
  # Download and cache the data
  # # 获取数据路径
  # path = tf.keras.utils.get_file(fname=os.path.basename(data_url), origin=data_url)


  df = pd.read_csv(data_url, names=CSV_COLUMN_NAMES, sep=',', header=0)

  # df[[]]

  df.info()

  print(df.describe().T)

  df.hist(bins=50, figsize=(20,15))
  plt.show()

  column_list = list(df.columns)[0:-1]

  sb.pairplot(df.loc[:, column_list], height=5)

  # plt.show()


  plt.savefig('d:/a.png')
  return df


# 读取数据,通过pandas 的 sample  api切分数据
def load_data_split_by_sample(y_name="Quality", train_fraction=0.7, seed=None):
  """Load the automobile data set and split it train/test and features/label.

  A description of the data is available at:
    hhttp://archive.ics.uci.edu/ml/machine-learning-databases/

  The data itself can be found at:
    http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the data set to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = load_data(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe()

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

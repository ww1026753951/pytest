import os
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import seaborn as sb

sb.set_context("notebook", font_scale=2.5)

from matplotlib import pyplot as plt
from natsort import natsorted

from sklearn.model_selection import train_test_split

import numpy as np


data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"


CSV_COLUMN_NAMES = ['Class', 'Alcohol', 'MalicAcid',
                    'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols',
                    'Proanthocyanins', 'ColorIntensity', 'Hue', 'DilutedWines', 'Proline']


# # 获取数据路径
# path = tf.keras.utils.get_file(fname=os.path.basename(data_url), origin=data_url)
#
# train = pd.read_csv(path, names=CSV_COLUMN_NAMES)
# # , random_state=42
# train, test = train_test_split(train, test_size=0.33)
#
# train.head()
#
# train_x, train_y = train, train.pop('class')
#
# test_x, test_y = test, test.pop('class')

def raw_dataframe():
  """Load the automobile data set as a pd.DataFrame."""
  # Download and cache the data
  CSV_COLUMN_NAMES = ['Class', 'Alcohol', 'MalicAcid',
                      'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols',
                      'Proanthocyanins', 'ColorIntensity', 'Hue', 'DilutedWines', 'Proline']

  # # 获取数据路径
  path = tf.keras.utils.get_file(fname=os.path.basename(data_url), origin=data_url)
  df = pd.read_csv(path, names=CSV_COLUMN_NAMES)

  return df


# 读取数据,通过pandas 的 sample  api切分数据
def load_data_split_by_sample(y_name="Class", train_fraction=0.7, seed=None):
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




# 读取数据,通过sklearn切分数据
def load_data_split_by_sklearn(y_name="Class", train_fraction=0.7, seed=None):
  # Load the raw data columns.
  data = raw_dataframe()

  # Delete rows with unknowns
  # data = data.dropna()

  # Shuffle the data
  # np.random.seed(seed)

  # Split the data into train/test subsets.
  # x_train = data.sample(frac=train_fraction, random_state=seed)
  # x_test = data.drop(x_train.index)

  # Extract the label from the features DataFrame.
  # y_train = x_train.pop(y_name)
  # y_test = x_test.pop(y_name)

  train, test = train_test_split(data, test_size=0.3)
  #
  # train.head()
  #
  x_train, y_train = train, train.pop('class')
  #
  x_test, y_test = test, test.pop('class')

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
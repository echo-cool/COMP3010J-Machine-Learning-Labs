# data analysis and preparing
import pandas as pd
import numpy as np
import random as rnd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree


# Read data file
df = pd.read_csv('kickstarter201801.csv')

# Split data into training, validation and test sets
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


def minmax_scaling(df, col):
    """Min-max scaling of the given column"""
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


# Print pie chart of the state of the projects
def print_pie_chart(df):
    labels = df['state'].value_counts(sort=True).index
    sizes = df['state'].value_counts(sort=True)
    colors = ["gold", "yellowgreen", "lightcoral", "lightskyblue"]
    explode = (0.05, 0.05, 0.05, 0.05)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,)
    plt.title('Project state')
    plt.show()
patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2, shadow=True, )
# # plot usd pledged histogram
# plt.hist(train['usd_pledged'], bins=100)
#
# # generate 1000 data points randomly drawn from an exponential distribution
# original_data = np.random.exponential(size = 1000)
#
# def minmax_scaling(df, col):
#     """Min-max scaling of the given column"""
#     min_val = df[col].min()
#     max_val = df[col].max()
#     df[col] = (df[col] - min_val) / (max_val - min_val)
#     return df
#
# # mix-max scale the data between 0 and 1
# scaled_data = minmax_scaling(original_data, col=[0])
#
# # plot both together to compare
# fig, ax=plt.subplots(1,2)
# sns.distplot(original_data, ax=ax[0])
# ax[0].set_title("Original Data")
# sns.distplot(scaled_data, ax=ax[1])
# ax[1].set_title("Scaled data")


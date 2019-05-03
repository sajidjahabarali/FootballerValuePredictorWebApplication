from __future__ import absolute_import, division, print_function
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


# store directory containing data in a variable
DATAPATH = 'C:\\Users\\sajid\\OneDrive\\Documents\\University\\Final Year Project\\Data\\fifa19\\dataNumerical.csv'


# import the dataset using pandas
def init_column_names():
    columnNames = ['Age',
                   'CountryWorldRankingPoints',
                   'Overall',
                   'Potential',
                   'ClubGoals',
                   'Value(€M)',
                   'Wage(€K)',
                   'Special',
                   'PreferredFoot',
                   'InternationalReputation',
                   'WeakFoot',
                   'SkillMoves',
                   'WorkRate',
                   'BodyType',
                   'Position',
                   'JerseyNumber',
                   'Height(cm)',
                   'Weight(lbs)',
                   'LS',
                   'ST',
                   'RS',
                   'LW',
                   'LF',
                   'CF',
                   'RF',
                   'RW',
                   'LAM',
                   'CAM',
                   'RAM',
                   'LM',
                   'LCM',
                   'CM',
                   'RCM',
                   'RM',
                   'LWB',
                   'LDM',
                   'CDM',
                   'RDM',
                   'RWB',
                   'LB',
                   'LCB',
                   'CB',
                   'RCB',
                   'RB',
                   'Crossing',
                   'Finishing',
                   'HeadingAccuracy',
                   'ShortPassing',
                   'Volleys',
                   'Dribbling',
                   'Curve',
                   'FKAccuracy',
                   'LongPassing',
                   'BallControl',
                   'Acceleration',
                   'SprintSpeed',
                   'Agility',
                   'Reactions',
                   'Balance',
                   'ShotPower',
                   'Jumping',
                   'Stamina',
                   'Strength',
                   'LongShots',
                   'Aggression',
                   'Interceptions',
                   'Positioning',
                   'Vision',
                   'Penalties',
                   'Composure',
                   'Marking',
                   'StandingTackle',
                   'SlidingTackle',
                   'GKDiving',
                   'GKHandling',
                   'GKKicking',
                   'GKPositioning',
                   'GKReflexes',
                   'ReleaseClause(€M)'
                   ]
    return columnNames


column_names = init_column_names()
raw_dataset = pd.read_csv(DATAPATH, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True,
                          encoding='latin-1')
print("raw dataset type: ", type(raw_dataset))
# print(type(raw_dataset.tail()))
pd.set_option('display.max_columns', 177)

# make a copy of the dataset to leave the original unaffected
dataset = raw_dataset.copy()

# clean the dataset by removing unknown values
dataset = dataset.dropna();
# printing the dataset
# print(dataset.tail())




def position_one_hot(dataset):
    position = dataset.pop("Position")
    dataset['CAM'] = (position == 'CAM') * 1.0
    dataset['CB'] = (position == 'CB') * 1.0
    dataset['CDM'] = (position == 'CDM') * 1.0
    dataset['CF'] = (position == 'CF') * 1.0
    dataset['CM'] = (position == 'CM') * 1.0
    dataset['GK'] = (position == 'GK') * 1.0
    dataset['LB'] = (position == 'LB') * 1.0
    dataset['LCB'] = (position == 'LCB') * 1.0
    dataset['LCM'] = (position == 'LCM') * 1.0
    dataset['LDM'] = (position == 'LDM') * 1.0
    dataset['LF'] = (position == 'LF') * 1.0
    dataset['LM'] = (position == 'LM') * 1.0
    dataset['LS'] = (position == 'LS') * 1.0
    dataset['LW'] = (position == 'LW') * 1.0
    dataset['LWB'] = (position == 'LWB') * 1.0
    dataset['RB'] = (position == 'RB') * 1.0
    dataset['RCB'] = (position == 'RCB') * 1.0
    dataset['RCM'] = (position == 'RCM') * 1.0
    dataset['RDM'] = (position == 'RDM') * 1.0
    dataset['RM'] = (position == 'RM') * 1.0
    dataset['RS'] = (position == 'RS') * 1.0
    dataset['RW'] = (position == 'RW') * 1.0
    dataset['RWB'] = (position == 'RWB') * 1.0
    dataset['ST'] = (position == 'ST') * 1.0

    return dataset


dataset = position_one_hot(dataset)

# print(dataset.tail())

# split the data into training and testing datasets.
training_data = dataset.sample(frac=0.8, random_state=0)
testing_data = dataset.drop(training_data.index)

# plot the data
# sns.pairplot(training_data[['Age',
#                             'Overall',
#                             'Potential',
#                             'Value(€M)',
#                             'Wage(€K)']], diag_kind="kde")
# plt.show()

# look at the statistics
train_stats = training_data.describe()
value_stats = train_stats.pop("Value(€M)")
train_stats = train_stats.transpose()
value_stats = value_stats.transpose()
# print(train_stats)
print(value_stats)

from __future__ import absolute_import, division, print_function
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# store directory containing data in a variable
DATAPATH = 'C:\\Users\\sajid\\OneDrive\\Documents\\University\\Final Year Project\\Data\\fifa19\\data.csv'

# import the dataset using pandas
columnNames = ['Name',
               'Age',
               'Nationality',
               'Overall',
               'Potential',
               'Club',
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

raw_dataset = pd.read_csv(DATAPATH, names=columnNames, na_values="?", comment='\t', sep=",", skipinitialspace=True, encoding='latin-1')
#pd.set_option('display.max_columns', 70)
# make a copy of the dataset to leave the original unaffected
dataset = raw_dataset.copy()
# dataset = dataset.dropna();
# printing the dataset
print(dataset.tail())



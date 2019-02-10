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
columnNames = ['Age',
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
pd.set_option('display.max_columns', 146)
# make a copy of the dataset to leave the original unaffected
dataset = raw_dataset.copy()
# dataset = dataset.dropna();
# printing the dataset
#print(dataset.tail())

# convert the categorical columns to a one-hot.
nationality = dataset.pop('Nationality')
dataset['Algeria'] = (nationality == 'Algeria') * 1.0
dataset['Argentina'] = (nationality == 'Argentina') * 1.0
dataset['Armenia'] = (nationality == 'Armenia') * 1.0
dataset['Australia'] = (nationality == 'Australia') * 1.0
dataset['Austria'] = (nationality == 'Austria') * 1.0
dataset['Belgium'] = (nationality == 'Belgium') * 1.0
dataset['Benin'] = (nationality == 'Benin') * 1.0
dataset['Bosnia Herzegovina'] = (nationality == 'Bosnia Herzegovina') * 1.0
dataset['Brazil'] = (nationality == 'Brazil') * 1.0
dataset['Cameroon'] = (nationality == 'Cameroon') * 1.0
dataset['Canada'] = (nationality == 'Canada') * 1.0
dataset['Chile'] = (nationality == 'Chile') * 1.0
dataset['Colombia'] = (nationality == 'Colombia') * 1.0
dataset['Croatia'] = (nationality == 'Croatia') * 1.0
dataset['Cyprus'] = (nationality == 'Cyprus') * 1.0
dataset['Czech Republic'] = (nationality == 'Czech Republic') * 1.0
dataset['Denmark'] = (nationality == 'Denmark') * 1.0
dataset['DR Congo'] = (nationality == 'DR Congo') * 1.0
dataset['Ecuador'] = (nationality == 'Ecuador') * 1.0
dataset['Egypt'] = (nationality == 'Egypt') * 1.0
dataset['England'] = (nationality == 'England') * 1.0
dataset['Equatorial Guinea'] = (nationality == 'Equatorial Guinea') * 1.0
dataset['France'] = (nationality == 'France') * 1.0
dataset['Gabon'] = (nationality == 'Gabon') * 1.0
dataset['Germany'] = (nationality == 'Germany') * 1.0
dataset['Ghana'] = (nationality == 'Ghana') * 1.0
dataset['Greece'] = (nationality == 'Greece') * 1.0
dataset['Guinea'] = (nationality == 'Guinea') * 1.0
dataset['Iceland'] = (nationality == 'Iceland') * 1.0
dataset['Iran'] = (nationality == 'Iran') * 1.0
dataset['Israel'] = (nationality == 'Israel') * 1.0
dataset['Italy'] = (nationality == 'Italy') * 1.0
dataset['Ivory Coast'] = (nationality == 'Ivory Coast') * 1.0
dataset['Jamaica'] = (nationality == 'Jamaica') * 1.0
dataset['Japan'] = (nationality == 'Japan') * 1.0
dataset['Kenya'] = (nationality == 'Kenya') * 1.0
dataset['Korea Republic'] = (nationality == 'Korea Republic') * 1.0
dataset['Mali'] = (nationality == 'Mali') * 1.0
dataset['Mexico'] = (nationality == 'Mexico') * 1.0
dataset['Montenegro'] = (nationality == 'Montenegro') * 1.0
dataset['Morocco'] = (nationality == 'Morocco') * 1.0
dataset['Netherlands'] = (nationality == 'Netherlands') * 1.0
dataset['New Zealand'] = (nationality == 'New Zealand') * 1.0
dataset['Nigeria'] = (nationality == 'Nigeria') * 1.0
dataset['Northern Ireland'] = (nationality == 'Northern Ireland') * 1.0
dataset['Norway'] = (nationality == 'Norway') * 1.0
dataset['Paraguay'] = (nationality == 'Paraguay') * 1.0
dataset['Philippines'] = (nationality == 'Philippines') * 1.0
dataset['Poland'] = (nationality == 'Poland') * 1.0
dataset['Portugal'] = (nationality == 'Portugal') * 1.0
dataset['Republic of Ireland'] = (nationality == 'Republic of Ireland') * 1.0
dataset['Romania'] = (nationality == 'Romania') * 1.0
dataset['Scotland'] = (nationality == 'Scotland') * 1.0
dataset['Senegal'] = (nationality == 'Senegal') * 1.0
dataset['Serbia'] = (nationality == 'Serbia') * 1.0
dataset['Slovakia'] = (nationality == 'Slovakia') * 1.0
dataset['Slovenia'] = (nationality == 'Slovenia') * 1.0
dataset['Spain'] = (nationality == 'Spain') * 1.0
dataset['Sweden'] = (nationality == 'Sweden') * 1.0
dataset['Switzerland'] = (nationality == 'Switzerland') * 1.0
dataset['Togo'] = (nationality == 'Togo') * 1.0
dataset['Tunisia'] = (nationality == 'Tunisia') * 1.0
dataset['Turkey'] = (nationality == 'Turkey') * 1.0
dataset['Ukraine'] = (nationality == 'Ukraine') * 1.0
dataset['United States'] = (nationality == 'United States') * 1.0
dataset['Uruguay'] = (nationality == 'Uruguay') * 1.0
dataset['Venezuela'] = (nationality == 'Venezuela') * 1.0
dataset['Wales'] = (nationality == 'Wales') * 1.0

print(dataset.tail())
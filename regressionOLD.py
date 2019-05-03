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
def init_column_names():
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
    return columnNames


column_names = init_column_names()
raw_dataset = pd.read_csv(DATAPATH, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True,
                          encoding='latin-1')

pd.set_option('display.max_columns', 177)

# make a copy of the dataset to leave the original unaffected
dataset = raw_dataset.copy()


# clean the dataset by removing unknown values
dataset = dataset.dropna();
# printing the dataset
print(dataset.tail())

# convert the nationality column to a one-hot.
def nationality_one_hot(dataset):
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

    return dataset
dataset = nationality_one_hot(dataset)


def club_one_hot(dataset):
    club = dataset.pop('Club')
    dataset['Arsenal'] = (club == 'Arsenal') * 1.0
    dataset['Bournemouth'] = (club == 'Bournemouth') * 1.0
    dataset['Brighton & Hove Albion'] = (club == 'Brighton & Hove Albion') * 1.0
    dataset['Burnley'] = (club == 'Burnley') * 1.0
    dataset['Cardiff City'] = (club == 'Cardiff City') * 1.0
    dataset['Chelsea'] = (club == 'Chelsea') * 1.0
    dataset['Crystal Palace'] = (club == 'Crystal Palace') * 1.0
    dataset['Everton'] = (club == 'Everton') * 1.0
    dataset['Fulham'] = (club == 'Fulham') * 1.0
    dataset['Huddersfield Town'] = (club == 'Huddersfield Town') * 1.0
    dataset['Leicester City'] = (club == 'Leicester City') * 1.0
    dataset['Liverpool'] = (club == 'Liverpool') * 1.0
    dataset['Manchester City'] = (club == 'Manchester City') * 1.0
    dataset['Manchester United'] = (club == 'Manchester United') * 1.0
    dataset['Newcastle United'] = (club == 'Newcastle United') * 1.0
    dataset['Southampton'] = (club == 'Southampton') * 1.0
    dataset['Tottenham Hotspur'] = (club == 'Tottenham Hotspur') * 1.0
    dataset['Watford'] = (club == 'Watford') * 1.0
    dataset['West Ham United'] = (club == 'West Ham United') * 1.0
    dataset['Wolverhampton Wanderers'] = (club == 'Wolverhampton Wanderers') * 1.0

    return dataset
dataset = club_one_hot(dataset)


def preferred_foot_one_hot(dataset):
    preferred_foot = dataset.pop("PreferredFoot")
    dataset['Right'] = (preferred_foot == 'Right') * 1.0
    dataset['Left'] = (preferred_foot == 'Left') * 1.0

    return dataset
dataset = preferred_foot_one_hot(dataset)


def work_rate_one_hot(dataset):
    work_rate = dataset.pop("WorkRate")
    dataset['High/ High'] = (work_rate == 'High/ High') * 1.0
    dataset['High/ Medium'] = (work_rate == 'High/ Medium') * 1.0
    dataset['High/ Low'] = (work_rate == 'High/ Low') * 1.0
    dataset['Medium/ High'] = (work_rate == 'Medium/ High') * 1.0
    dataset['Medium/ Medium'] = (work_rate == 'Medium/ Medium') * 1.0
    dataset['Medium/ Low'] = (work_rate == 'Medium/ High') * 1.0
    dataset['Low/ High'] = (work_rate == 'Low/ High') * 1.0
    dataset['Low/ Medium'] = (work_rate == 'Low/ Medium') * 1.0

    return dataset
dataset = work_rate_one_hot(dataset)


def body_type_one_hot(dataset):
    body_type = dataset.pop('BodyType')
    dataset['Lean'] = (body_type == 'Lean') * 1.0
    dataset['Normal'] = (body_type == 'Normal') * 1.0
    dataset['Stocky'] = (body_type == 'Stocky') * 1.0
    dataset['PLAYER_BODY_TYPE_25'] = (body_type == 'PLAYER_BODY_TYPE_25') * 1.0
    dataset['Shaqiri'] = (body_type == 'Shaqiri') * 1.0

    return dataset
dataset = body_type_one_hot(dataset)


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
sns.pairplot(training_data[['Age',
                            'Overall',
                            'Potential',
                            'Value(€M)',
                            'Wage(€K)']], diag_kind="kde")
# plt.show()

# look at the statistics
train_stats = training_data.describe()
train_stats.pop("Value(€M)")
train_stats = train_stats.transpose()
# print(train_stats)

# separate the label (which we are trying to predict) from the rest of the features.
train_labels = training_data.pop('Value(€M)')
test_labels = testing_data.pop('Value(€M)')


# create method to normalize data passed into it used the mean and standard deviation.
def normalize(data):
    return (data - train_stats['mean']) / train_stats['std']


# normalize the training and testing data and store the normalized data in variables.
normTrainingData = normalize(training_data)
normTestingData = normalize(testing_data)


# new method which can be called to build the model
def buildModel():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# build the model and store it in a variable.
model = buildModel()

# inspect the model.
print(model.summary())

# test the model on a batch of 10 examples from the training data.
example = normTrainingData[:5]
exampleResult = model.predict(example)
print(exampleResult)

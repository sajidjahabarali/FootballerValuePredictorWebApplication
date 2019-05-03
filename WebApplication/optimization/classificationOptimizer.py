# import tensorflow library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
import datetime

# import numpy library
import numpy as np

# import pandas library
import pandas as pd

# import seaborn library
import seaborn as sns

# import pyplot library
import matplotlib.pyplot as plt

# store directory containing data in a variable
# DATAPATH = 'C:\\Users\\sajid\\OneDrive\\Documents\\University\\Final Year Project\\Data\\fifa19\\dataNumerical.csv'
# VALUES = 'C:\\Users\\sajid\\OneDrive\\Documents\\University\\Final Year Project\\Data\\fifa19\\player values for classification.csv'

currentDT = datetime.datetime.now()
print (str(currentDT))

DATAPATH = '../../Data/fifa19/dataNumerical.csv'
VALUES = '../../Data/fifa19/player values for classification.csv'

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
                   'S# printSpeed',
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


class_names = ['0-100K',
               '101-500K',
               '500K-1M',
               '1-5M',
               '5-10M',
               '10-25M',
               '25-50M',
               '50-75M',
               '75-100M',
               '100M+']

# print('==================================================')
# print('READ DATA CSV')
column_names = init_column_names()
raw_dataset = pd.read_csv(DATAPATH, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True,
                          encoding='latin-1'
                          )

# print('==================================================')
# print('READ PLAYER VALUES CSV')
playerValues = pd.read_csv(VALUES, names=['values'], na_values="?", comment='\t', sep=",", skipinitialspace=True,
                           encoding='latin-1'
                           )

# pd.set_option('display.max_columns', 177)
# pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 5)

# make a copy of the dataset to leave the original unaffected
dataset = raw_dataset.copy()

# clean the dataset by removing unknown values
dataset = dataset.dropna();
# # print(type(dataset))

# create classification labels for the dataset
classification_labels = playerValues.values
# # print(classification_labels)
# # print(type(classification_labels))


# create empty array then append the different classes into the array.
class_labels_array = []
# print('==================================================')
# print('CREATING CLASS LABELS')
for x in classification_labels:
    if x <= 0.1:
        class_labels_array.append(0)
    elif x < 0.5:
        class_labels_array.append(1)
    elif x < 1:
        class_labels_array.append(2)
    elif x < 5:
        class_labels_array.append(3)
    elif x < 10:
        class_labels_array.append(4)
    elif x < 25:
        class_labels_array.append(5)
    elif x < 50:
        class_labels_array.append(6)
    elif x < 75:
        class_labels_array.append(7)
    elif x < 100:
        class_labels_array.append(8)
    else:
        class_labels_array.append(9)

# # print(class_labels_array)
class_labels = pd.DataFrame(data=class_labels_array, columns=['class_labels'])
# # print(class_labels)
# # print("end of class labels")
# # printing the dataset
# # print(dataset.tail())


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

# convert the categorical position column into a one-hot.
dataset = position_one_hot(dataset)

# # print(dataset.tail())

# # print(type(dataset))

# split the data into training and testing datasets.
training_data = dataset.sample(frac=0.8, random_state=0)
# print("=============================")
# print("TRAINING DATA EXAMPLES")
# print(training_data.tail())
testing_data = dataset.drop(training_data.index)
# print("=============================")
# print("TESTING DATA EXAMPLES")
# print(testing_data.tail())

# remove value we are trying to predict from the dataset.
training_data.pop('Value(€M)')
testing_data.pop('Value(€M)')

# split the labels into a training and testing labels.
training_labels = class_labels.sample(frac=0.8, random_state=0)
# print("=============================")
# print("TRAINING LABELS")
# print(training_labels.tail())
testing_labels = class_labels.drop(training_labels.index)
# print("=============================")
# print("TESTING LABELS")
# print(testing_labels.tail())



# new method which can be called to build the model
def buildModel(learningRate):
    model = keras.Sequential([
        # layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
        # layers.Dense(64, activation=tf.nn.relu),
        # layers.Dense(10, activation=tf.nn.softmax)
        keras.layers.Flatten(input_shape=[len(training_data.keys())]),
        # keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(64, activation=tf.nn.sigmoid),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  # optimizer=tf.train.AdamOptimizer(lr=0.0001),
                  optimizer=tf.keras.optimizers.Adam(learningRate),
                  metrics=['accuracy'])
    return model


# build the model and store it in a variable.
EPOCHS = [350, 400, 450, 500, 550]
# EPOCHS = [150, 200]
learningRates = [0.01, 0.001, 0.0001, 0.00001]
# learningRates = [0.1, 0.01]
batch_sizes = [8, 16, 32]
# batch_sizes = [16, 32]


bestEpoch = -1
bestLR = -1
bestBatchSize = -1
bestAcc = -1

# print('==================================================')
# print('TRAINING')
# train the model
for current_epoch in EPOCHS:
    for current_lr in learningRates:
        for current_batch_size in batch_sizes:
            model = buildModel(current_lr)
            model.fit(training_data, training_labels, epochs=current_epoch, batch_size=current_batch_size, verbose=0)

            # print('==================================================')
            # print('ACCURACY TESTING')
            # check accuracy
            test_loss, test_accuracy = model.evaluate(testing_data, testing_labels)
            # print('Test accuracy:', test_accuracy)

            if test_accuracy>bestAcc:
                bestAcc = test_accuracy
                bestLR = current_lr
                bestEpoch = current_epoch
                bestBatchSize = current_batch_size

print('BEST ACCURACY: ', bestAcc)
print('BEST EPOCH: ', bestEpoch)
print('BEST LR: ', bestLR)
print('BEST BATCH SIZE: ', bestBatchSize)


currentDT = datetime.datetime.now()
print (str(currentDT))
# make prediction for first player in dataset
# predictions = model.predict(testing_data)
# print('==================================================')
# print('CLASS PROBABILITIES')
# print(predictions[0])
# print('==================================================')
# print('CLASSIFICATION')

# for x in range(len(predictions)):
# for x in range(10):
    # # print the strongest class from the prediction
    # print('PREDICTION: ', class_names[np.argmax(predictions[x])])

    # # print the correct classification of the player that the prediction was made on to see if classification was correct.
    # # print(testing_labels.tail())
    # print('ACTUAL: ', class_names[int(testing_labels.iloc[x]['class_labels'])])
    # print()

# print('==================================================')
# print('END')

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
train_stats.pop("Value(€M)")
train_stats = train_stats.transpose()
# print(train_stats)

# separate the label (which we are trying to predict) from the rest of the features.
train_labels = training_data.pop('Value(€M)')
testingLabels = testing_data.pop('Value(€M)')


# create method to normalize data passed into it used the mean and standard deviation.
def normalize(data):
    return (data - train_stats['mean']) / train_stats['std']


# normalize the training and testing data and store the normalized data in variables.
normTrainingData = normalize(training_data)
normTestingData = normalize(testing_data)


# new method which can be called to build the model
def buildModel(lossParam, optimizerParam, activationParam):
    model = keras.Sequential([
        layers.Dense(64, activation=activationParam, input_shape=[len(training_data.keys())]),
        layers.Dense(64, activation=activationParam),
        layers.Dense(1)
    ])

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss=lossParam,
                  optimizer=optimizerParam,
                  metrics=['mae', 'mse', 'accuracy'])
    return model


# build the model and store it in a variable.
model = buildModel('mae', tf.keras.optimizers.RMSprop(0.001), tf.nn.relu)

# inspect the model.
# print(model.summary())

# test the model on a batch of 10 examples from the training data.
example = normTrainingData[:10]
exampleResult = model.predict(example)
# print(exampleResult)


# Display training progress by printing a single dot for each completed epoch
class DotPrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


# set the number of epochs to train the model for.

EPOCHS = 1500

# train the model and store the training and validation accuracy in a history object.
# history = model.fit(
#     normTrainingData, train_labels,
#     epochs=EPOCHS, validation_split=0.2, verbose=0,
    #callbacks=[DotPrinter()]
    # )

# Use the stats stores in the history object to visualize the models training progress.
#hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch
#print()
#print(hist.tail())


# create method to plot the model on a graph
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Validation Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Validation Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# call the plot method to plot the stats stored in the history object on a graph
# plot_history(history)

lossParams = ['mse'
              #, 'mae'
             ]
batchSizes = [32, 64, 128]
activationParams = [tf.nn.relu, tf.nn.sigmoid]

paramLabels = ["epochs", "lossParam", "activationParam", "optimizerParam", "learningRateParam", "batchSize"]
lowestMSEParams = ["epochs", "lossParam", "activationParam", "optimizerParam", "learningRateParam", "batchSize"]
lowestMSE = 100000;
for epochParam in range(4,10):
    EPOCHS = 500 * epochParam
    for lossParam in lossParams:
        for activationParam in activationParams:
            for learningRateParam in range(3, 5):
                optimizerParams = [tf.keras.optimizers.RMSprop(lr=10**(learningRateParam*-1)), tf.keras.optimizers.SGD(lr=10**(learningRateParam*-1))]
                for optimizerParam in optimizerParams:
                    for batchSize in batchSizes:
                        # Use EarlyStopping callback to test a training condition to stop training when the validation score stops improving.

                        model = buildModel(lossParam, optimizerParam, activationParam)
                        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

                        history = model.fit(normTrainingData, train_labels, batch_size=batchSize, epochs=EPOCHS,
                                            validation_split=0.2, verbose=0, callbacks=[early_stop,
                                                                                        # DotPrinter()
                                                                                        ])

                        # call the plot method to plot the stats stored in the history object on a graph
                        # plot_history(history)

                        # use testing set to see how well model generalizes.
                        loss, mae, mse, accuracy = model.evaluate(normTestingData, testingLabels, verbose=0)
                        print("--------------------------------------------------------------------")
                        #print(model.metrics_names)
                        print("epochs: " + str(EPOCHS))
                        print("loss: " + str(lossParam))
                        if(activationParam == tf.nn.relu):
                            print("activation: relu")
                        else:
                            print("activation: sigmoid")


                        print("Loss of Testing Set:€{:5.2f} M".format(loss))
                        print("Mean Absolute Error of Testing Set:€{:5.2f} M".format(mae))
                        print("Mean Squared Error of Testing Set:€{:5.2f} M".format(mse))
                        print("Accuracy of Testing Set:{:5.2f} ".format(accuracy))

                        if(mse<lowestMSE):
                            lowestMSE = mse
                            if (activationParam == tf.nn.relu):
                                activationParam = "relu"
                            else:
                                activationParam = "sigmoid"

                            if (optimizerParam == tf.keras.optimizers.RMSprop(lr=10 ** (learningRateParam * -1))):
                                optimizerParam = "RMSprop"
                            else:
                                optimizerParam = "SGD"

                            lowestMSEParams = [str(EPOCHS), str(lossParams), activationParam, optimizerParam, 10 ** (learningRateParam * -1), batchSize]


print()
print()
print()
print()
print("==================================================")
print("Lowest MSE = " + str(lowestMSE))
for param in range(1,6):
    print(paramLabels[param] + ": " + str(lowestMSEParams[param]))


# Use the testing dataset to predict player values
# testingPredictions = model.predict(normTestingData).flatten()
#
# plt.scatter(testingLabels, testingPredictions)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0, plt.xlim()[1]])
# plt.ylim([0, plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()
#
# # plot the error distribution
# error = testingPredictions - testingLabels
# plt.hist(error, bins=25)
# plt.xlabel("Prediction Error [MPG]")
# _ = plt.ylabel("Count")
# plt.show()



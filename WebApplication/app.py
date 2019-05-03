import os
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from uuid import uuid4
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from flask import Flask, request, render_template, send_from_directory, jsonify

app = Flask(__name__, static_url_path='/static')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    filename = ""
    target = os.path.join(APP_ROOT, '../FootballerValuePredictor')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)

    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("directory not created: {}".format(target))


    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("file name: {}".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("file inputted:", filename)
        print ("Save to:", destination)
        upload.save(destination)

    # return send_from_directory("files", filename, as_attachment=True)
    return render_template("player-value-predictor.html", file_name=filename)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory("data", filename)


def get_training_and_testing_datasets():
    raw_dataset = get_data()
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()

    dataset = position_one_hot(dataset)
    training_data = dataset.sample(frac=0.8, random_state=0)
    testing_data = dataset.drop(training_data.index)

    return training_data, testing_data

def normalize(data, train_stats):
    return (data - train_stats['mean']) / train_stats['std']

def build_regression_model(lossParam, optimizerParam, activationParam, training_data):
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

class DotPrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        # print('.', end='')

@app.route('/regression', methods=['POST'])
def regression():
    training_data, testing_data = get_training_and_testing_datasets()

    train_stats = training_data.describe()
    train_stats.pop("Value(€M)")
    train_stats = train_stats.transpose()

    train_labels = training_data.pop('Value(€M)')
    testingLabels = testing_data.pop('Value(€M)')

    normTrainingData = normalize(training_data, train_stats)
    normTestingData = normalize(testing_data, train_stats)


    EPOCHS = 1500
    try:
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model('rmodel.h5')
    except:
        print('LOAD MODEL ERROR')
        model = build_regression_model('mse', tf.keras.optimizers.SGD(lr=0.001), tf.nn.sigmoid, training_data)
        model.save('rmodel.h5')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

        history = model.fit(normTrainingData, train_labels, batch_size=32, epochs=EPOCHS,
                            validation_split=0.2, verbose=1, callbacks=[early_stop,
                                                                        DotPrinter()
                                                                        ])
        model.save('rmodel.h5')
    loss, mae, mse, accuracy = model.evaluate(normTestingData, testingLabels, verbose=0)
    testingPredictions = model.predict(normTestingData).flatten()
    print('ACCURACY: ', accuracy)
    print('MSE: ', mse)
    # predict single player value
    PLAYER_DATA_PATH = 'data.csv'

    column_names = init_column_names()
    input_player_data = pd.read_csv(PLAYER_DATA_PATH, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True,
                              encoding='latin-1')

    input_player_data = input_player_data.dropna()
    input_player_data = position_one_hot(input_player_data)
    actual_value = input_player_data.pop('Value(€M)')
    # print("FIRST")
    # print(input_player_data.tail())

    # input_player_data = pd.DataFrame([[27,1726,85,86,63,45,180,2060,1,3,4,4,8,1,"ST",9,175.26,161,85,85,85,84,86,86,86,84,84,84,84,82,77,77,77,82,65,63,63,63,65,62,57,57,57,62,64,88,73,80,81,85,74,71,59,83,81,84,85,85,85,83,76,79,73,75,70,42,86,76,84,85,29,42,30,11,6,9,5,6,88.9]], columns=column_names)
    # input_player_data = input_player_data.dropna()
    # input_player_data = position_one_hot(input_player_data)
    # actual_value = input_player_data.pop('Value(€M)')

    # print("actual value: ", actual_value.values[0])
    # print("SECOND")
    # print(input_player_data.tail())

    prediction = model.predict(input_player_data).flatten()

    # print("single player prediction: ", prediction)

    response = {
    # 'prediction': str(type(data))
    'prediction': str(round(float(str(prediction)[1:(len(str(prediction))-1)]), 1)),
    'actual_value': str(actual_value.values[0])
    }
    return jsonify(response)


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



@app.route('/classification', methods=['POST'])
def classification():
    training_data, testing_data = get_training_and_testing_datasets()
    VALUES = 'player values for classification.csv'

    class_names = ['0-100K',
                   '100-500K',
                   '500K-1M',
                   '1-5M',
                   '5-10M',
                   '10-25M',
                   '25-50M',
                   '50-75M',
                   '75-100M',
                   '100M+']

    playerValues = pd.read_csv(VALUES, names=['values'], na_values="?", comment='\t', sep=",", skipinitialspace=True,
                               encoding='latin-1'
                               )

    # dataset = raw_dataset.copy()
    # dataset = dataset.dropna();
    classification_labels = playerValues.values

    # response = {
    #     'prediction': str(classification_labels)
    # }
    # return jsonify(response)

    class_labels_array = []

    for x in classification_labels:
        if float(x) <= 0.1:
            class_labels_array.append(0)
        elif float(x) < 0.5:
            class_labels_array.append(1)
        elif float(x) < 1:
            class_labels_array.append(2)
        elif float(x) < 5:
            class_labels_array.append(3)
        elif float(x) < 10:
            class_labels_array.append(4)
        elif float(x) < 25:
            class_labels_array.append(5)
        elif float(x) < 50:
            class_labels_array.append(6)
        elif float(x) < 75:
            class_labels_array.append(7)
        elif float(x) < 100:
            class_labels_array.append(8)
        else:
            class_labels_array.append(9)

    class_labels = pd.DataFrame(data=class_labels_array, columns=['class_labels'])

    # response = {
    #     'prediction': (str(len(classification_labels)), str(len(class_labels)))
    # }
    # return jsonify(response)

    # dataset = position_one_hot(dataset)
    # training_data = dataset.sample(frac=0.8, random_state=0)
    # testing_data = dataset.drop(training_data.index)

    training_data.pop('Value(€M)')
    testing_data.pop('Value(€M)')

    training_labels = class_labels.sample(frac=0.8, random_state=0)
    testing_labels = class_labels.drop(training_labels.index)

    # response = {
    #     'prediction': (str(training_labels.tail()), str(testing_labels.tail()))
    # }
    # return jsonify(response)
    try:
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model= load_model('cmodel.h5')
    except:
        print('LOAD MODEL ERROR')
        model = build_classification_model(training_data)
        model.fit(training_data, training_labels, epochs=1375, batch_size=16)
        model.save('cmodel.h5')

    test_loss, mae, mse, test_accuracy = model.evaluate(testing_data, testing_labels)
    print('ACCURACY: ', test_accuracy)
    print('MSE: ', mse)

    predictionValues = model.predict(testing_data)
    print(predictionValues[0])
    prediction = class_names[np.argmax(predictionValues[0])]



    response = {
        'prediction': prediction
    }
    return jsonify(response)

def build_classification_model(training_data):
    model = keras.Sequential([
        # layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
        # layers.Dense(64, activation=tf.nn.relu),
        # layers.Dense(10, activation=tf.nn.softmax)
        keras.layers.Flatten(input_shape=[len(training_data.keys())]),
        # keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
        keras.layers.Dense(256, activation=tf.nn.sigmoid),
        keras.layers.Dense(256, activation=tf.nn.sigmoid),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  # optimizer=tf.train.AdamOptimizer(lr=0.0001),
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['mae', 'mse', 'accuracy'])
    return model

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

def get_data():
    DATAPATH = 'dataNumerical.csv'
    column_names = init_column_names()
    raw_dataset = pd.read_csv(DATAPATH, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True,
                              encoding='latin-1')
    return raw_dataset

if __name__ == "__main__":
    app.run(port=5000, debug=True)

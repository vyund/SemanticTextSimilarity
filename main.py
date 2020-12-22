from utils import load_data
from metrics import pearson_corr, spearman_corr
from data_preprocess import build_embedding, prepare_data

from malstm import MaLSTM
from bilstm import BiLSTM
from cnnlstm import CNNLSTM
from bicnnlstm import BiCNNLSTM

import argparse
import numpy as np
from time import time

import matplotlib.pyplot as plt

from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping

# run training loop for model with early stopping (if specified)
def train_model(model, train_data, val_data, early_stopping, batch_size=64, num_epochs=25):
    print('Training ')
    start_time = time()

    X_train = train_data[0]
    y_train = train_data[1]
    X_val = val_data[0]
    y_val = val_data[1]

    if early_stopping:
        es = EarlyStopping(monitor='val_loss', patience=2)
        trained_model = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[es])
    else:
        trained_model = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val))

    time_elapsed = time() - start_time
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return trained_model

# evaluate trained model on test data
def evaluate_model(model, test_data):
    X_test = test_data[0]
    y_test = test_data[1]

    scores = model.evaluate(X_test, y_test)

    y_preds = model.predict(X_test)
    pearson_r = pearson_corr(y_test, y_preds)
    spearman_r = spearman_corr(y_test, y_preds)

    print('MSE: {}'.format(round(scores[0], 4)))
    print('Pearson Correlation: {}'.format(round(pearson_r, 4)))
    print('Spearman Correlation: {}'.format(round(spearman_r, 4)))

# plot training history of model for specified metric
def plot_training(trained_model, model_name, metric):
    plt.plot(trained_model.history['{}'.format(metric)])
    plt.plot(trained_model.history['val_{}'.format(metric)])
    plt.title('{} Model {}'.format(model_name, metric))
    plt.ylabel('{}'.format(metric))
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# initialize MaLSTM model and run experiments based on user input (default params as described in original papers)
def run_malstm(embedding_info, train, test, val, dist_func='manhattan', early_stopping=True, plot=True):
    num_units = 50
    malstm = MaLSTM(embedding_info, num_units, dist_func)
    adadelta = Adadelta(learning_rate=0.01, clipnorm=1.25)

    malstm.compile(loss='mse', optimizer=adadelta, metrics=['mse'])

    batch_size = 64
    num_epochs = 25

    trained_model = train_model(malstm, train, val, early_stopping, batch_size, num_epochs)
    evaluate_model(malstm, test)

    if plot:
        plot_training(trained_model, 'MaLSTM', 'loss')

# initialize BiLSTM model and run experiments based on user input (default params as described in original papers)
def run_bilstm(embedding_info, train, test, val, dist_func='cosine', early_stopping=True, plot=True):
    num_units = 64
    bilstm = BiLSTM(embedding_info, num_units, dist_func)
    adam = Adam(learning_rate=0.00001)

    bilstm.compile(loss='mse', optimizer=adam, metrics=['mse'])

    batch_size = 64
    num_epochs = 100

    trained_model = train_model(bilstm, train, val, early_stopping, batch_size, num_epochs)
    evaluate_model(bilstm, test)

    if plot:
        plot_training(trained_model, 'BiLSTM', 'loss')

# initialize CNN-LSTM model and run experiments based on user input (default params as described in original papers)
def run_cnnlstm(embedding_info, train, test, val, local_context=3, dist_func='manhattan', early_stopping=True, plot=True):
    num_units = 50
    cnnlstm = CNNLSTM(embedding_info, num_units, local_context, dist_func)
    adadelta = Adadelta(learning_rate=0.01, clipnorm=1.25)

    cnnlstm.compile(loss='mse', optimizer=adadelta, metrics=['mse'])

    batch_size = 64
    num_epochs = 50

    trained_model = train_model(cnnlstm, train, val, early_stopping, batch_size, num_epochs)
    evaluate_model(cnnlstm, test)

    if plot:
        plot_training(trained_model, 'CNN-LSTM', 'loss')

# initialize BiCNN-LSTM model and run experiments based on user input
def run_bicnnlstm(embedding_info, train, test, val, local_context=3, dist_func='cosine', early_stopping=True, plot=True):
    num_units = 64
    bicnnlstm = BiCNNLSTM(embedding_info, num_units, local_context, dist_func)
    adam = Adam(learning_rate=0.00001)

    bicnnlstm.compile(loss='mse', optimizer=adam, metrics=['mse'])

    batch_size = 64
    num_epochs = 100

    trained_model = train_model(bicnnlstm, train, val, early_stopping, batch_size, num_epochs)
    evaluate_model(bicnnlstm, test)

    if plot:
        plot_training(trained_model, 'BiCNN-LSTM', 'loss')

if __name__ == '__main__':
    # default params
    model = 'malstm'
    data = 'SICK'
    distance = 'manhattan'
    early_stop = True
    plot = True  

    # user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help='select model architecture (Default = malstm)')
    parser.add_argument('--data', '-d', help='select dataset (Default = SICK)')
    parser.add_argument('--function', '-f', help='select distance function')
    parser.add_argument('-localcontext', '-l', help='for CNN models ONLY - select local context parameter ')
    parser.add_argument('-earlystop', '-e', help='1=True / 0=False for using early stopping (Default = True)')
    parser.add_argument('-plot', '-p', help='1=True / 0=False for plotting performance (Default = True)')

    args = parser.parse_args()
    
    if args.model:
        model = args.model
        # default params based on model
        if model == 'malstm':
            distance = 'manhattan'
        elif model == 'bilstm':
            distance = 'cosine'
        elif model == 'cnnlstm':
            distance = 'manhattan'
            local_context = 3
        elif model == 'bicnnlstm':
            distance='cosine'
            local_context = 3
    if args.data:
        data = args.data
    if args.function:
        distance = args.function
    if args.localcontext:
        local_context = int(args.localcontext)
    if args.earlystop:
        if args.earlystop == 1:
            early_stop = True
        else:
            early_stop = False
    if args.plot:
        if args.plot == 1:
            plot = True
        else:
            plot = False

    # load in data
    train_data_dir = '../../data/{}_train.txt'.format(data)
    test_data_dir = '../../data/{}_test.txt'.format(data)

    train_data = load_data(train_data_dir)
    test_data = load_data(test_data_dir)
    
    # generate embedding matrix
    embedding_dim = 300
    embedding_matrix, train_data, test_data = build_embedding(train_data, test_data)

    # process data
    train, val, test, max_seq_length = prepare_data(train_data, test_data)

    embedding_info = [embedding_dim, embedding_matrix, max_seq_length]

    # output experiment info
    print('Using {} model with {} function...'.format(model, distance))

    # model select
    if model == 'malstm':
        run_malstm(embedding_info, train, test, val, distance, early_stop, plot)
    elif model == 'bilstm':
        run_bilstm(embedding_info, train, test, val, distance, early_stop, plot)
    elif model == 'cnnlstm':
        run_cnnlstm(embedding_info, train, test, val, local_context, distance, early_stop, plot)
    elif model == 'bicnnlstm':
        run_bicnnlstm(embedding_info, train, test, val, local_context, distance, early_stop, plot)
import pandas as pd
import numpy as np
import time
from keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

from model import Model
from utils import draw_plots


def calc():
    # Wczytanie danych
    df = pd.read_csv("data/gallstone.data", sep=',')

    # Podział kolumn
    X = df.iloc[:, 1:].values # cechy
    y = df.iloc[:, 0].values # etykiety

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    epochs = 170
    test_size = 0.2

    # Model basic
    model_basic = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_basic = model_basic.learn(X, y, test_size=test_size)
    print("Model basic:", time.time() - start_time, "s")

    # Model z dropout
    model_dropout = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_dropout = model_dropout.learn(X, y, test_size=test_size)
    print("Model dropout:", time.time() - start_time, "s")

    # Model z regularyzacją L1
    model_l1 = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_l1 = model_l1.learn(X, y, test_size=test_size)
    print("Model l1:", time.time() - start_time, "s")

    # Model z regularyzacją L2
    model_l2 = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_l2 = model_l2.learn(X, y, test_size=test_size)
    print("Model l2:", time.time() - start_time, "s")

    # Model z EarlyStopping
    model_early_stopping = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_early_stopping = model_early_stopping.learn(X, y, test_size=test_size, early_stopping=10)
    print("Model early_stopping:", time.time() - start_time, "s")

    # Model simplified
    model_simplified = Model([
        Input(shape=(n_features,)),
        Dense(10, activation='relu'),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_simplified = model_simplified.learn(X, y, test_size=test_size)
    print("Model simplified:", time.time() - start_time, "s")

    # Model augment
    model_augment = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_augment = model_augment.learn(X, y, test_size=test_size, augment=True)
    print("Model augment:", time.time() - start_time, "s")

    # Model z L1 + L2 + Dropout
    model_l1_l2_dropout = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-5)),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_l1_l2_dropout = model_l1_l2_dropout.learn(X, y, test_size=test_size)
    print("Model l1_l2_dropout:", time.time() - start_time, "s")

    # Model z L2 + Dropout 
    model_l2_dropout = Model([
        Input(shape=(n_features,)),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(5e-4)),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ], epochs)

    start_time = time.time()
    history_l2_dropout = model_l2_dropout.learn(X, y, test_size=test_size)
    print("Model l2_dropout:", time.time() - start_time, "s")

    # Wykresy
    draw_plots(history_basic, history_dropout, 'basic', 'dropout', 'gallstone_last') 
    draw_plots(history_basic, history_l1, 'basic', 'l1', 'gallstone_last') 
    draw_plots(history_basic, history_l2, 'basic', 'l2', 'gallstone_last') 
    draw_plots(history_basic, history_early_stopping, 'basic', 'early_stopping', 'gallstone_last') 
    draw_plots(history_basic, history_simplified, 'basic', 'simplified', 'gallstone_last') 
    draw_plots(history_basic, history_augment, 'basic', 'augment', 'gallstone_last') 
    draw_plots(history_basic, history_l1_l2_dropout, 'basic', 'l1_l2_dropout', 'gallstone_last')
    draw_plots(history_basic, history_l2_dropout, 'basic', 'l2_dropout', 'gallstone_last') 
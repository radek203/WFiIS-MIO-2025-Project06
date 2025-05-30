import pandas as pd
import time
from keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

from model import Model
from utils import draw_plots


def calc():
    # Wczytanie danych
    df = pd.read_csv("data/yeast.data", sep=r"\s+", header=None)

    # Podział kolumn
    X = df.iloc[:, 1:9].values # cechy
    y = df.iloc[:, 9].values # etykiety

    model_basic = Model([
            Input(shape=(8,)), # 8 cech
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax') #10 kategorii
        ], 200)

    start_time = time.time()
    history_basic = model_basic.learn(X, y)
    print("Model basic: ", time.time() - start_time, "s")

    model_dropout = Model([
            Input(shape=(8,)), # 8 cech
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax') #10 kategorii
        ], 200)

    start_time = time.time()
    history_dropout = model_dropout.learn(X, y)
    print("Model dropout: ", time.time() - start_time, "s")

    model_l1 = Model([
            Input(shape=(8,)), # 8 cech
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, kernel_regularizer=regularizers.l1(0.01), activation='softmax') #10 kategorii
        ], 200)

    start_time = time.time()
    history_l1 = model_l1.learn(X, y)
    print("Model l1: ", time.time() - start_time, "s")

    model_l2 = Model([
            Input(shape=(8,)), # 8 cech
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, kernel_regularizer=regularizers.l2(0.01), activation='softmax') #10 kategorii
        ], 200)

    start_time = time.time()
    history_l2 = model_l2.learn(X, y)
    print("Model l2: ", time.time() - start_time, "s")

    start_time = time.time()
    history_early_stopping = model_basic.learn(X, y, early_stopping=5)
    print("Model early_stopping: ", time.time() - start_time, "s")

    model_simplified = Model([
            Input(shape=(8,)), # 8 cech
            Dense(10, activation='relu'),
            Dense(10, activation='relu'),
            Dense(10, activation='softmax') #10 kategorii
        ], 200)

    start_time = time.time()
    history_simplified = model_simplified.learn(X, y)
    print("Model simplified: ", time.time() - start_time, "s")

    start_time = time.time()
    history_augment = model_basic.learn(X, y, augment=True)
    print("Model augment: ", time.time() - start_time, "s")

    model_l1_l2_dropout = Model([
            Input(shape=(8,)), # 8 cech
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(10, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), activation='softmax') #10 kategorii
        ], 200)

    start_time = time.time()
    history_l1_l2_dropout = model_l1_l2_dropout.learn(X, y)
    print("Model l1_l2_dropout: ", time.time() - start_time, "s")

    draw_plots(history_basic, history_dropout, 'basic', 'dropout', 'yeast')
    draw_plots(history_basic, history_l1, 'basic', 'l1', 'yeast')
    draw_plots(history_basic, history_l2, 'basic', 'l2', 'yeast')
    draw_plots(history_basic, history_early_stopping, 'basic', 'early stopping', 'yeast')
    draw_plots(history_basic, history_simplified, 'basic', 'simplified', 'yeast')
    draw_plots(history_basic, history_augment, 'basic', 'augment', 'yeast')
    draw_plots(history_basic, history_l1_l2_dropout, 'basic', 'l1_l2_dropout', 'yeast')

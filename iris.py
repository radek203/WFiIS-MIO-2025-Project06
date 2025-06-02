import pandas as pd
import time
from keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

from model import ModelForSmallDatasets
from utils import draw_plots


def calc():
    # Wczytanie danych
    df = pd.read_csv("data/iris.data", header=None)
    df.dropna(inplace=True)

    # Podział kolumn
    X = df.iloc[:, 0:4].values # cechy
    y = df.iloc[:, 4].values # etykiety

    # Model basic
    model_basic = ModelForSmallDatasets([
            Input(shape=(4,)), # 4 cechy
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax') #3 kategorii
        ], 200)


    start_time = time.time()
    history_basic = model_basic.learn(X, y)
    print("Model basic: ", time.time() - start_time, "s")

    # Model z dropout
    model_dropout = ModelForSmallDatasets([
            Input(shape=(4,)), # 4 cechy
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax') #3 kategorii
        ], 200)

    start_time = time.time()
    history_dropout = model_dropout.learn(X, y)
    print("Model dropout: ", time.time() - start_time, "s")

    # Model z EarlyStopping
    start_time = time.time()
    history_early_stopping = model_basic.learn(X, y, early_stopping=15)
    print("Model early_stopping: ", time.time() - start_time, "s")

    # Model z regularyzacją L1
    model_l1 = ModelForSmallDatasets([
            Input(shape=(4,)), # 4 cechy
            Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01),),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01),),
            Dense(3, activation='softmax') #3 kategorii
        ], 200)

    start_time = time.time()
    history_l1 = model_l1.learn(X, y)
    print("Model l1: ", time.time() - start_time, "s")

    # Model z regularyzacją L2
    model_l2 = ModelForSmallDatasets([
            Input(shape=(4,)), # 4 cechy
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01),),
            Dense(3, activation='softmax') #3 kategorii
        ], 200)

    start_time = time.time()
    history_l2 = model_l2.learn(X, y)
    print("Model l2: ", time.time() - start_time, "s")

    # Model simplified
    model_simplified = ModelForSmallDatasets([
            Input(shape=(4,)), # 4 cechy
            Dense(10, activation='relu'),
            Dense(10, activation='relu'),
            Dense(3, activation='softmax') #3 kategorii
        ], 200)

    start_time = time.time()
    history_simplified = model_simplified.learn(X, y)
    print("Model simplified: ", time.time() - start_time, "s")

    # Model augment
    start_time = time.time()
    history_augment = model_basic.learn(X, y, augment=True)
    print("Model augment: ", time.time() - start_time, "s")

    # Model l1_l2_dropout
    model_l1_l2_dropout = ModelForSmallDatasets([
            Input(shape=(4,)), # 4 cechy
            Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax') #3 kategorii
        ], 200)

    start_time = time.time()
    history_l1_l2_dropout = model_l1_l2_dropout.learn(X, y)
    print("Model l1_l2_dropout: ", time.time() - start_time, "s")

    # Wykresy
    draw_plots(history_basic, history_dropout, 'basic', 'dropout', 'iris', output_dir='results/iris')
    draw_plots(history_basic, history_l1, 'basic', 'l1', 'iris', output_dir='results/iris')
    draw_plots(history_basic, history_l2, 'basic', 'l2', 'iris', output_dir='results/iris')
    draw_plots(history_basic, history_early_stopping, 'basic', 'early stopping', 'iris', output_dir='results/iris')
    draw_plots(history_basic, history_simplified, 'basic', 'simplified', 'iris', output_dir='results/iris')
    draw_plots(history_basic, history_augment, 'basic', 'augment', 'iris', output_dir='results/iris')
    draw_plots(history_basic, history_l1_l2_dropout, 'basic', 'l1_l2_dropout', 'iris', output_dir='results/iris')

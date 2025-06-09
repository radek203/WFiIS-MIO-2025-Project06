import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from utils import augment_data

class Model:

    def __init__(self, structure, epochs):
        self.structure = structure
        self.epochs = epochs

    def learn(self, X, y, test_size, early_stopping=-1, augment=False):
        # Zakodowanie etykiet
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_cat = to_categorical(y_encoded)

        # Skalowanie danych
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Podział na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_cat, test_size=test_size, stratify=y_encoded, random_state=42
        )

        if augment:
            X_aug = augment_data(X_train)
            y_aug = y_train.copy()

            # Łączenie danych oryginalnych i zaugmentowanych
            X_train = np.vstack([X_train, X_aug])
            y_train = np.vstack([y_train, y_aug])

        # Budowa modelu
        model = Sequential(self.structure)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='macro')]
        )

        # Trening
        if early_stopping >= 1:
            es = EarlyStopping(
                monitor='val_f1_score', # 'val_accuracy', 'val_loss'
                patience=early_stopping # liczba epok bez poprawy
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=32,
                verbose=0,
                callbacks=[es]
            )
        else:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=32,
                verbose=0
            )

        # Obliczenie różnicy accuracy między treningiem a walidacją
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        accuracy_diff = train_acc - val_acc

        print(f"Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}, Difference (train - val): {accuracy_diff:.4f}")

        return history

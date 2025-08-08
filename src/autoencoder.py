# core/autoencoder.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    bottleneck = Dense(16, activation="relu", name="bottleneck")(encoded)

    decoded = Dense(32, activation="relu")(bottleneck)
    decoded = Dense(64, activation="relu")(decoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder

def train_autoencoder(X):
    autoencoder, encoder = build_autoencoder(X.shape[1])
    autoencoder.fit(X, X, epochs=20, batch_size=64, shuffle=True, verbose=0)
    return encoder

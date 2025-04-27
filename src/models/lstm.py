
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(X_train, y_train):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

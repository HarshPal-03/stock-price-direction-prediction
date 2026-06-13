import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

# LOAD DATA
data = pd.read_csv(r'C:\Users\harsh\OneDrive\Desktop\Stock price\stocks.csv\EW-MAX.csv')

print("Columns:")
print(data.columns.tolist())

# Convert date
data['Date'] = pd.to_datetime(data['Date'])

# Sort by date
data = data.sort_values('Date')

# TECHNICAL INDICATORS
data['SMA_10'] = data['Close'].rolling(window=10).mean()

data['SMA_20'] = data['Close'].rolling(window=20).mean()

data['EMA_10'] = data['Close'].ewm(span=10,adjust=False).mean()

data['Returns'] = data['Close'].pct_change()
data = data.dropna()

# VISUAL STOCK TREND
plt.figure(figsize=(14, 6))
plt.plot(
    data['Date'],
    data['Close'],
    label='Close Price')

plt.plot(
    data['Date'],
    data['SMA_10'],
    label='SMA 10')

plt.plot(
    data['Date'],
    data['SMA_20'],
    label='SMA 20')

plt.plot(
    data['Date'],
    data['EMA_10'],
    label='EMA 10')

plt.title('Stock Trend Analysis')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# FEATURES
features = [
    'Close',
    'Volume',
    'SMA_10',
    'SMA_20',
    'EMA_10',
    'Returns'
]
data_features = data[features].values

# SCALING
scaler = MinMaxScaler()
data_features_scaled = scaler.fit_transform(data_features)

# TARGET
data['target'] = (
    data['Close'].shift(-1)
    > data['Close']
).astype(int)

data = data[:-1]
data_features_scaled = data_features_scaled[:-1]
target = data['target'].values

# CREATE SEQUENCES
def create_sequences(data,target,seq_length):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
        y.append(target[i + seq_length])

    return np.array(X), np.array(y)


SEQ_LENGTH = 20
X, y = create_sequences(data_features_scaled,target,SEQ_LENGTH)

# TRAIN TEST SPLIT
split_index = int(len(X) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# LSTM INPUT SHAPE

X_train_lstm = X_train.reshape(
    (X_train.shape[0],SEQ_LENGTH,len(features))
)

X_test_lstm = X_test.reshape(
    (X_test.shape[0],SEQ_LENGTH,len(features))
)

# LSTM MODEL


lstm_model = Sequential([LSTM(50,return_sequences=True,input_shape=(SEQ_LENGTH,len(features))),
            LSTM(50),Dense(1,activation='sigmoid')])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_lstm = lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)
y_pred_lstm = lstm_model.predict(X_test_lstm)

# NEURAL NETWORK
nn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

history_nn = nn_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
    )

y_pred_nn = nn_model.predict(X_test)

# LOGISTIC REGRESSION
logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(
    X_train,
    y_train
)

y_pred_logistic = (
    logistic_model.predict(X_test)
)

# CONVERT TO BINARY
y_pred_lstm_binary = (
    y_pred_lstm.flatten() > 0.5).astype(int)
y_pred_nn_binary = (
    y_pred_nn.flatten() > 0.5).astype(int)

# METRICS

accuracy_lstm = accuracy_score(
    y_test,
    y_pred_lstm_binary
)

accuracy_nn = accuracy_score(
    y_test,
    y_pred_nn_binary
)

accuracy_logistic = accuracy_score(
    y_test,
    y_pred_logistic
)

f1_lstm = f1_score(
    y_test,
    y_pred_lstm_binary
)

f1_nn = f1_score(
    y_test,
    y_pred_nn_binary
)

f1_logistic = f1_score(
    y_test,
    y_pred_logistic)

print(f"LSTM Accuracy: {accuracy_lstm:.4f}")
print(f"NN Accuracy: {accuracy_nn:.4f}")
print(f"Logistic Accuracy: {accuracy_logistic:.4f}")

# LOSS CURVES

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('LSTM Loss')
plt.legend(['Train','Validation'])
plt.subplot(1,2,2)
plt.plot(history_nn.history['loss'])
plt.plot(history_nn.history['val_loss'])
plt.title('NN Loss')
plt.legend(['Train','Validation'])
plt.show()

# ACTUAL VS PREDICTED
plt.figure(figsize=(14,6))
plt.plot(y_test[:100],label='Actual')
plt.plot(y_pred_lstm_binary[:100],label='LSTM Prediction')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# CONFUSION MATRIX
cm = confusion_matrix(y_test,y_pred_logistic)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('Logistic Regression Confusion Matrix')
plt.show()

# MODEL COMPARISON
models = [
    'LSTM',
    'NN',
    'Logistic'
]

accuracies = [
    accuracy_lstm,
    accuracy_nn,
    accuracy_logistic
]

plt.figure(figsize=(8,5))
sns.barplot(x=models,y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- USE CURRENT WORKING DIRECTORY ---
base_path = os.getcwd()  # Automatically uses the folder where this script is run
os.makedirs(base_path, exist_ok=True)

DATA_PATH = os.path.join(base_path, "Per_hour_dataSet.csv")  # Place your CSV here
MODEL_PATH = os.path.join(base_path, "LSTM_best_model.h5")
RESULT_TEST1_PATH = os.path.join(base_path, "output_load_forecasting_result_test1.txt")
RESULT_TEST2_PATH = os.path.join(base_path, "output_load_forecasting_result_test2.txt")

sequence_length = 50
batch_size = 64
epochs = 200

# --- DATA LOADING ---
df_raw = pd.read_csv(DATA_PATH, header=None)
hourly_load = df_raw.iloc[:, 1].astype(float).values  # Assuming second column is the load

# --- FUNCTION TO CONVERT SERIES TO 2D SUPERVISED MATRIX ---
def convertSeriesToMatrix(series, seq_length):
    matrix = []
    for i in range(len(series) - seq_length + 1):
        matrix.append(series[i:i + seq_length])
    return np.array(matrix)

matrix_load = convertSeriesToMatrix(hourly_load, sequence_length)

# --- SCALING ---
scaler = StandardScaler()
matrix_load_scaled = scaler.fit_transform(matrix_load)

# --- SPLIT DATA INTO TRAIN, VALIDATION, TEST1, TEST2 SETS ---
n_samples = matrix_load_scaled.shape[0]
train_end = int(round(0.7 * n_samples))
val_end = int(round(0.8 * n_samples))
test1_end = int(round(0.9 * n_samples))

train_set = matrix_load_scaled[:train_end, :]
val_set = matrix_load_scaled[train_end:val_end, :]
test1_set = matrix_load_scaled[val_end:test1_end, :]
test2_set = matrix_load_scaled[test1_end:, :]

# --- SPLIT INPUT / OUTPUT ---
def splitInputOutput(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

X_train, y_train = splitInputOutput(train_set)
X_val, y_val = splitInputOutput(val_set)
X_test1, y_test1 = splitInputOutput(test1_set)
X_test2, y_test2 = splitInputOutput(test2_set)

# --- RESHAPE FOR LSTM (samples, time_steps, features) ---
def reshape_for_lstm(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

X_train = reshape_for_lstm(X_train)
X_val = reshape_for_lstm(X_val)
X_test1 = reshape_for_lstm(X_test1)
X_test2 = reshape_for_lstm(X_test2)

# --- MODEL BUILDING ---
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(150, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='linear'),
])

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])

# --- CALLBACKS INCLUDING MODEL CHECKPOINTING ---
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

# --- TRAINING ---
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping, checkpoint]
)

# --- LOAD BEST MODEL AFTER TRAINING ---
model = load_model(MODEL_PATH)
print(f"Best model loaded from {MODEL_PATH}")

# --- EVALUATION ON TEST SETS ---
loss_test1, mae_test1 = model.evaluate(X_test1, y_test1, verbose=1)
loss_test2, mae_test2 = model.evaluate(X_test2, y_test2, verbose=1)

print(f"Test Set 1 - MSE: {loss_test1:.4f}, MAE: {mae_test1:.4f}")
print(f"Test Set 2 - MSE: {loss_test2:.4f}, MAE: {mae_test2:.4f}")

# --- PREDICTIONS ---
pred_test1_scaled = model.predict(X_test1)
pred_test2_scaled = model.predict(X_test2)

# --- INVERSE TRANSFORM TO ORIGINAL SCALE ---
def invert_scaling(X_scaled, y_scaled):
    combined = np.hstack([X_scaled, y_scaled])
    inverted = scaler.inverse_transform(combined)
    return inverted[:, -1]  # return last column (target)

true_test1 = invert_scaling(X_test1.reshape((X_test1.shape[0], -1)), y_test1.reshape(-1, 1))
pred_test1 = invert_scaling(X_test1.reshape((X_test1.shape[0], -1)), pred_test1_scaled)

true_test2 = invert_scaling(X_test2.reshape((X_test2.shape[0], -1)), y_test2.reshape(-1, 1))
pred_test2 = invert_scaling(X_test2.reshape((X_test2.shape[0], -1)), pred_test2_scaled)

# --- ADDITIONAL METRICS ---
print("Test Set 1 - Additional Metrics:")
print("MAE:", mean_absolute_error(true_test1, pred_test1))
print("R² Score:", r2_score(true_test1, pred_test1))

print("Test Set 2 - Additional Metrics:")
print("MAE:", mean_absolute_error(true_test2, pred_test2))
print("R² Score:", r2_score(true_test2, pred_test2))

# --- PLOT RESULTS ---
plt.figure(figsize=(12, 6))
plt.plot(true_test1, label='True Load')
plt.plot(pred_test1, label='Predicted Load')
plt.title('Test Set 1 - True vs Predicted Load')
plt.xlabel('Time Step')
plt.ylabel('Load')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(true_test2, label='True Load')
plt.plot(pred_test2, label='Predicted Load')
plt.title('Test Set 2 - True vs Predicted Load')
plt.xlabel('Time Step')
plt.ylabel('Load')
plt.legend()
plt.grid(True)
plt.show()

# --- SAVE PREDICTION RESULTS ---
np.savetxt(RESULT_TEST1_PATH, np.column_stack((pred_test1, true_test1)),
           header='Predicted\tTrue', fmt='%.6f')
np.savetxt(RESULT_TEST2_PATH, np.column_stack((pred_test2, true_test2)),
           header='Predicted\tTrue', fmt='%.6f')

print(f"Prediction results saved to:\n - {RESULT_TEST1_PATH}\n - {RESULT_TEST2_PATH}")

# Modified for Module 5 Kubernetes practice - Nov 23 2025

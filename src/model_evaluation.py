import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load the data
df = pd.read_csv('ML Quick Projects for ISRO/Martian Dust Storm Prediction/data/processed/engineered_features_overlapping.csv')

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

# Select features for the model
features = ['ls', 'min_temp', 'max_temp', 'pressure', 'wind_speed', 'Area (square km)', 
            'pressure_diff', 'temp_diff', 'temp_range', 'distance_from_equator', 
            'pressure_lag7', 'avg_temp_lag7', 'wind_speed_lag7', 'Area (square km)_lag7']

X = df[features]
y = df['dust_storm_intensity']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input data for LSTM (samples, time steps, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Rebuild the model architecture
def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build and load weights
model = build_model((X_reshaped.shape[1], X_reshaped.shape[2]))
model.load_weights('ML Quick Projects for ISRO/Martian Dust Storm Prediction/models/best_model.h5')

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_index, test_index in tscv.split(X_reshaped):
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    cv_scores.append(mae)

print(f"Cross-validation MAE scores: {cv_scores}")
print(f"Mean CV MAE: {np.mean(cv_scores)}")

# Evaluate on the entire dataset
y_pred = model.predict(X_reshaped).flatten()
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Create a dataframe with results
results_df = df[['terrestrial_date', 'sol', 'ls']].copy()
results_df['actual_intensity'] = y
results_df['predicted_intensity'] = y_pred
results_df['dust_storm_occurrence'] = results_df['predicted_intensity'].apply(lambda x: 'Yes' if x > 0 else 'No')

# Export results to CSV
results_df.to_csv('ML Quick Projects for ISRO/Martian Dust Storm Prediction/results/dust_storm_predictions.csv', index=False)

print("Model evaluation completed. Results exported to dust_storm_predictions.csv")
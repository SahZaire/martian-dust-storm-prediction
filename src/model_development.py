import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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

# Load the trained model
model = load_model('ML Quick Projects for ISRO/Martian Dust Storm Prediction/models/best_model.h5')

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

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Dust Storm Intensity")
plt.ylabel("Predicted Dust Storm Intensity")
plt.title("Actual vs Predicted Dust Storm Intensity")
plt.savefig('ML Quick Projects for ISRO/Martian Dust Storm Prediction/results/actual_vs_predicted.png')
plt.close()

# Create a dataframe with results
results_df = df[['terrestrial_date', 'sol', 'ls']].copy()
results_df['actual_intensity'] = y
results_df['predicted_intensity'] = y_pred
results_df['dust_storm_occurrence'] = results_df['predicted_intensity'].apply(lambda x: 'Yes' if x > 0 else 'No')

# Export results to CSV
results_df.to_csv('ML Quick Projects for ISRO/Martian Dust Storm Prediction/results/dust_storm_predictions.csv', index=False)

print("Model evaluation completed. Results exported to dust_storm_predictions.csv")

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, edgecolor='black')
plt.title('Distribution of Dust Storm Intensity')
plt.xlabel('Dust Storm Intensity')
plt.ylabel('Frequency')
plt.savefig('ML Quick Projects for ISRO/Martian Dust Storm Prediction/results/dust_storm_intensity_distribution.png')
plt.close()

print(f"\nDust Storm Intensity Statistics:")
print(y.describe())

# Check for any extreme outliers
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
outliers = y[(y < lower_bound) | (y > upper_bound)]
print(f"\nNumber of outliers: {len(outliers)}")
print(f"Percentage of outliers: {(len(outliers) / len(y)) * 100:.2f}%")
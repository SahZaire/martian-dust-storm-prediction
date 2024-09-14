import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_solar_longitude_features(df):
    ls_column = 'Ls' if 'Ls' in df.columns else 'ls'
    df['Ls_rad'] = np.radians(df[ls_column])
    df['sin_Ls'] = np.sin(df['Ls_rad'])
    df['cos_Ls'] = np.cos(df['Ls_rad'])
    return df

def estimate_wind_speed(df):
    df['pressure_diff'] = df['pressure'].diff().abs()
    df['temp_diff'] = (df['max_temp'] - df['min_temp']).abs()
    
    df['pressure_diff_norm'] = (df['pressure_diff'] - df['pressure_diff'].min()) / (df['pressure_diff'].max() - df['pressure_diff'].min())
    df['temp_diff_norm'] = (df['temp_diff'] - df['temp_diff'].min()) / (df['temp_diff'].max() - df['temp_diff'].min())
    
    df['wind_speed'] = 2 + 28 * (df['pressure_diff_norm'] + df['temp_diff_norm']) / 2
    
    return df

def estimate_dust_storm_intensity(df):
    df['temp_range'] = df['max_temp'] - df['min_temp']
    df['dust_storm_intensity'] = pd.cut(df['temp_range'] * df['pressure'] / 1000, 
                                        bins=[0, 1000, 2000, 3000, np.inf], 
                                        labels=[0, 1, 2, 3])
    return df

def estimate_atmo_opacity(df):
    df['atmo_opacity'] = pd.cut(df['temp_range'] * df['pressure'] / 1000, 
                                bins=[0, 1000, 2000, 3000, np.inf], 
                                labels=['Clear', 'Cloudy', 'Dusty', 'Very Dusty'])
    return df

def calculate_atmospheric_features(df):
    df['avg_temp'] = (df['min_temp'] + df['max_temp']) / 2
    df = estimate_wind_speed(df)
    df = estimate_dust_storm_intensity(df)
    df = estimate_atmo_opacity(df)
    return df

def calculate_geographical_features(df):
    if 'Centroid latitude' in df.columns:
        df['distance_from_equator'] = abs(df['Centroid latitude'])
    else:
        temp_variation = df['max_temp'] - df['min_temp']
        df['distance_from_equator'] = 90 * (temp_variation - temp_variation.min()) / (temp_variation.max() - temp_variation.min())
    
    df['northern_hemisphere'] = (df['distance_from_equator'] > 0).astype(int)
    df['southern_hemisphere'] = (df['distance_from_equator'] <= 0).astype(int)
    return df

def engineer_features(df):
    df = calculate_solar_longitude_features(df)
    df = calculate_atmospheric_features(df)
    df = calculate_geographical_features(df)
    
    lag_features = ['pressure', 'avg_temp', 'wind_speed']
    if 'Area (square km)' in df.columns:
        lag_features.append('Area (square km)')
    
    for feature in lag_features:
        df[f'{feature}_lag1'] = df.groupby('Mars Year')[feature].shift(1)
        df[f'{feature}_lag7'] = df.groupby('Mars Year')[feature].shift(7)
        df[f'{feature}_7sol_std'] = df.groupby('Mars Year')[feature].rolling(window=7, center=True).std().reset_index(0, drop=True)
    
    return df

def main():
    df_overlapping = load_data('ML Quick Projects for ISRO/Martian Dust Storm Prediction/data/processed/overlapping_data.csv')
    df_overlapping_engineered = engineer_features(df_overlapping)
    df_overlapping_engineered.to_csv('ML Quick Projects for ISRO/Martian Dust Storm Prediction/data/processed/engineered_features_overlapping.csv', index=False)
    logger.info("Engineered features for overlapping data saved to 'data/processed/engineered_features_overlapping.csv'")

    df_weather = load_data('ML Quick Projects for ISRO/Martian Dust Storm Prediction/data/raw/mars_weather.csv')
    df_weather['Mars Year'] = df_weather['martian_year']
    df_weather_engineered = engineer_features(df_weather)
    
    for col in df_overlapping_engineered.columns:
        if col not in df_weather_engineered.columns:
            if col in ['Member ID', 'Sequence ID', 'Confidence interval', 'Missing data']:
                df_weather_engineered[col] = np.nan
            elif col in ['Centroid longitude', 'Centroid latitude', 'Maximum latitude', 'Minimum latitude']:
                df_weather_engineered[col] = df_weather_engineered['distance_from_equator']
            elif col == 'Area (square km)':
                df_weather_engineered[col] = 0
            else:
                df_weather_engineered[col] = df_weather_engineered[col.split('_')[0]] if '_' in col else 0
    
    df_weather_engineered.to_csv('ML Quick Projects for ISRO/Martian Dust Storm Prediction/data/processed/engineered_features_weather.csv', index=False)
    logger.info("Engineered features for Mars weather data saved to 'data/processed/engineered_features_weather.csv'")

    logger.info("\nEngineered Overlapping Dataset Statistics:")
    logger.info(f"Total records: {len(df_overlapping_engineered)}")
    logger.info(f"Number of features: {df_overlapping_engineered.shape[1]}")
    
    logger.info("\nEngineered Mars Weather Dataset Statistics:")
    logger.info(f"Total records: {len(df_weather_engineered)}")
    logger.info(f"Number of features: {df_weather_engineered.shape[1]}")
    
    logger.info("\nFeature list:")
    logger.info(df_overlapping_engineered.columns.tolist())

if __name__ == "__main__":
    main()
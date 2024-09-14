import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_mdad_data(file_path):
    df = pd.read_csv(file_path)
    df['Mars Year'] = df['Mars Year'].astype(int)
    df['Ls'] = df['Ls'].astype(int)
    return df

def load_mars_weather_data(file_path):
    df = pd.read_csv(file_path)
    df['martian_year'] = df['martian_year'].astype(int)
    df['ls'] = df['ls'].astype(int)
    return df

# Merging datasets based on Martian Year and Ls
def merge_datasets(mdad_df, mars_weather_df):
    merged_df = pd.merge(mdad_df, mars_weather_df, 
                         left_on=['Mars Year', 'Ls'], 
                         right_on=['martian_year', 'ls'], 
                         how='outer', 
                         suffixes=('_mdad', '_weather'))

    overlapping_df = pd.merge(mdad_df, mars_weather_df, 
                              left_on=['Mars Year', 'Ls'], 
                              right_on=['martian_year', 'ls'], 
                              how='inner')

    return merged_df, overlapping_df

def preprocess_data(df):
    logger.info(f"Preprocessing data. Input shape: {df.shape}")
    
    if df.empty:
        logger.warning("Empty DataFrame passed to preprocess_data. Returning empty DataFrame.")
        return df
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Numeric columns: {numeric_columns.tolist()}")
    
    imputer = SimpleImputer(strategy='mean')
    
    try:
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        logger.info("Imputation completed successfully.")
    except ValueError as e:
        logger.error(f"Error during imputation: {str(e)}")
        logger.info("Columns with all NaN values:")
        for col in numeric_columns:
            if df[col].isna().all():
                logger.info(f"- {col}")
    
    logger.info(f"Preprocessing completed. Output shape: {df.shape}")
    return df

def main():
    try:
        mdad_df = load_mdad_data(r'ML Quick Projects for ISRO\Martian Dust Storm Prediction\data\raw\MDAD.csv')
        mars_weather_df = load_mars_weather_data(r'ML Quick Projects for ISRO\Martian Dust Storm Prediction\data\raw\mars_weather.csv')

        logger.info(f"MDAD data shape: {mdad_df.shape}")
        logger.info(f"Mars Weather data shape: {mars_weather_df.shape}")

        if mdad_df is None or mars_weather_df is None:
            logger.error("Data loading failed. Please check the error messages above.")
            return

        # MDAD-only dataset
        mdad_only = preprocess_data(mdad_df)
        mdad_only.to_csv(r'ML Quick Projects for ISRO\Martian Dust Storm Prediction\data\processed\mdad_only.csv', index=False)
        logger.info("MDAD-only data saved to 'data/processed/mdad_only.csv'")

        # Merged datasets
        merged_data, overlapping_data = merge_datasets(mdad_df, mars_weather_df)
        logger.info(f"Merged data shape: {merged_data.shape}")
        logger.info(f"Overlapping data shape: {overlapping_data.shape}")
        
        merged_data = preprocess_data(merged_data)
        overlapping_data = preprocess_data(overlapping_data)
        
        logger.info(f"Preprocessed merged data shape: {merged_data.shape}")
        logger.info(f"Preprocessed overlapping data shape: {overlapping_data.shape}")
        
        merged_data.to_csv(r'ML Quick Projects for ISRO\Martian Dust Storm Prediction\data\processed\combined_data.csv', index=False)
        logger.info("Combined data saved to 'data/processed/combined_data.csv'")
        
        overlapping_data.to_csv(r'ML Quick Projects for ISRO\Martian Dust Storm Prediction\data\processed\overlapping_data.csv', index=False)
        logger.info("Overlapping data saved to 'data/processed/overlapping_data.csv'")

        logger.info("\nMDAD-only Dataset Statistics:")
        logger.info(f"Total records: {len(mdad_only)}")
        logger.info(f"Martian Year range: {mdad_only['Mars Year'].min()} to {mdad_only['Mars Year'].max()}")

        logger.info("\nCombined Dataset Statistics:")
        logger.info(f"Total records: {len(merged_data)}")
        logger.info(f"Martian Year range: {merged_data['Mars Year'].min()} to {merged_data['Mars Year'].max()}")

        logger.info("\nOverlapping Dataset Statistics:")
        logger.info(f"Total records: {len(overlapping_data)}")
        logger.info(f"Martian Year range: {overlapping_data['Mars Year'].min()} to {overlapping_data['Mars Year'].max()}")

        logger.info("\nMissing data percentage in combined dataset:")
        logger.info(merged_data.isnull().sum() * 100 / len(merged_data))

        logger.info("\nMissing data percentage in overlapping dataset:")
        logger.info(overlapping_data.isnull().sum() * 100 / len(overlapping_data))

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        logger.error("Please check your data files and ensure they contain the expected columns.")

if __name__ == "__main__":
    main()
# Martian Dust Storm Prediction with Time Series Analysis

This project predicts dust storms on Mars using historical atmospheric data from various sources, including NASA's MAVEN mission, the Mars Atmospheric Data Assimilation (MDAD) dataset, Mars Weather data, and the REMS (Rover Environmental Monitoring Station) Mars dataset. Time series forecasting techniques are applied to predict dust storm occurrences based on key atmospheric indicators.

## Project Overview

The goal of this project is to develop models that predict Martian dust storms using data from NASA’s Planetary Data System (PDS) and other publicly available Martian weather datasets. The project compares different time series models (LSTM, Prophet, ARIMA) and establishes prediction thresholds to detect dust storm occurrences based on atmospheric patterns.

### Datasets Used:
- **MDAD**: Mars Dust Activity Database
- **MAVEN**: Neutral Gas and Ion Mass Spectrometer data
- **Mars Weather**: Public weather data from Mars Science Laboratory
- **REMS Mars**: Rover Environmental Monitoring Station dataset

### Factors for Dust Storm Estimation:
The dust storm prediction is based on the following factors and thresholds:

1. **Wind Speed**:
   - > 17 m/s: +1 intensity
   - > 25 m/s: +1 additional intensity
2. **Pressure Change**:
   - > 1 mb change: +1 intensity
3. **Temperature Change**:
   - > 10K change: +1 intensity
4. **Storm Area**:
   - > 50,000 sq km: +1 intensity

These thresholds are derived from the Mars Fact Sheet and assumptions about significant changes in atmospheric conditions.

## Future Work

### 1. **Model Fine-Tuning and Error Reduction**
   - Further optimization of the current models (LSTM, Prophet, ARIMA) is needed to improve accuracy and reduce both the training and validation errors. Techniques like hyperparameter tuning, feature selection, and advanced data augmentation strategies could be applied to minimize the error rates beyond the current values of 9.25% for training and 3% for validation.

### 2. **Incorporating Graphical Data for Enhanced Prediction**
   - The model could be enhanced by incorporating additional datasets, such as Martian surface imagery and weather reports, to gain better insights into dust storm patterns. Data sources like:
     - [MSSS Mars Weather Reports](https://www.msss.com/msss_images/subject/weather_reports.html)
     - [Mars Raw Images from MSL](https://prod.mars.jpllab.net/msl/multimedia/raw-images/?order=sol+desc%2Cinstrument_sort+asc%2Csample_type_sort+asc%2C+date_taken+desc&per_page=100&page=1&mission=msl)
     
   These sources provide visual and graphical data that can be combined with atmospheric data to create more comprehensive storm detection models. By deploying the model on graphical data, the prediction system could not only rely on numerical data but also analyze images to detect potential storm activity.

### 3. **Deployment**
   - Once the model is refined and incorporates graphical data, it can be deployed as a real-time prediction tool using APIs and cloud-based platforms. A basic web interface can be developed to allow users to input new data or view storm predictions on Martian surface maps.

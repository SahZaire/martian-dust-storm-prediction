from datetime import datetime

# Reference date for Martian Year 31 (August 7, 2012)
start_date = datetime(2012, 8, 7)
martian_year_start = 31
sols_per_martian_year = 668.6
earth_days_per_sol = 1.027

def calculate_martian_year(terrestrial_date):
    current_date = datetime.strptime(terrestrial_date, '%Y-%m-%d')
    
    earth_days_diff = (current_date - start_date).days
    
    sols_passed = earth_days_diff / earth_days_per_sol
    
    martian_year = martian_year_start + (sols_passed // sols_per_martian_year)
    
    return int(martian_year)

import pandas as pd

file_path = r'ML Quick Projects for ISRO\Martian Dust Storm Prediction\data\raw\mars-weather.csv'  
mars_data = pd.read_csv(file_path)

mars_data['martian_year'] = mars_data['terrestrial_date'].apply(calculate_martian_year)

mars_data.to_csv('updated_mars_weather.csv', index=False)

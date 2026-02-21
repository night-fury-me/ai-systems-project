import pandas as pd
import re

def load_word_lists(file_path):
    df = pd.read_csv(file_path)
    
    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    df = df.dropna(subset=['city_ascii', 'population']) 
    df = df[df['population'] >= 1e5]
    
    diacritic_pattern = r'[^\w]'
    mask = df['city_ascii'].str.contains(diacritic_pattern, regex=True, na=False)
    df   = df[~mask]
    
    df   = df.drop_duplicates(subset=["city_ascii"])
    tz_mask     = df['iso2'].fillna('').str.strip() == 'TZ'
    tz_cities   = df[tz_mask]['city_ascii'].str.upper().tolist()
    non_tz_cities = df[~tz_mask]['city_ascii'].str.upper().tolist()

    return tz_cities, non_tz_cities
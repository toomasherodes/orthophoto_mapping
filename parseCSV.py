import pandas as pd
import os
from shapely import wkb

def wkb_to_coords(w):
    if pd.isna(w):
        return None
    geometry = wkb.loads(bytes.fromhex(str(w)))
    return geometry.wkt

def parseCSV(sample_csv_path, parsed_csv_path):
    current_dir = os.path.dirname(__file__)
    f = open(os.path.join(current_dir, sample_csv_path), encoding='utf-8')
    df = pd.read_csv(f, encoding='utf-8', delimiter=';')
    parsed_geom = df.geometry.apply(wkb_to_coords)
    parsed_df = pd.DataFrame({
        'wkb': df.geometry,
        'wkt': parsed_geom
    })
    parsed_df.to_csv(os.path.join(current_dir, parsed_csv_path))

parseCSV('data/csv/data_sample.csv', 'data/csv/parsed_data.csv')

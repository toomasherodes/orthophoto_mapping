import pandas as pd
import os
from shapely import wkb
from shapely.geometry import Polygon

def translate_coord(coords, offset_x=6470000, offset_y=655000, scale=5): # scale is pixels per coord
    # default offsets and scale is for square 54752 with scale 1:10000
    x = int((coords[0]-offset_x)*scale)
    y = int((coords[1]-offset_y)*scale)
    return (y, 25000-x)


def wkb_to_coords(w):
    if pd.isna(w):
        return None
    geometry = wkb.loads(bytes.fromhex(str(w)))
    return geometry.wkt

def wkb_to_polygons(w):
    if pd.isna(w):
        return None
    geometry = wkb.loads(bytes.fromhex(str(w)))
    if not isinstance(geometry, Polygon):
        return None
    polygon = [translate_coord((x, y)) for y, x in geometry.exterior.coords]
    if (polygon[0][0] > 26000 or polygon[0][0] < -1000):
        return None
    if (polygon[0][1] > 26000 or polygon[0][1] < -1000):
        return None
    return polygon



def get_polygons_from_csv(sample_csv_path):
    current_dir = os.path.dirname(__file__)
    f = open(os.path.join(current_dir, sample_csv_path), encoding='utf-8')
    df = pd.read_csv(f, encoding='utf-8', delimiter=';', low_memory=False)
    
    keywords = ['HOONE', 'RAJATIS', 'HOONERAJ']
    pattern = '|'.join(keywords)
    filtered_df = df[df['nahtus'].str.contains(pattern, case=False, na=False)]
    
    parsed_geom = filtered_df.geometry.apply(wkb_to_polygons).dropna()
    print("finished parsing polygons")
    return parsed_geom.tolist()




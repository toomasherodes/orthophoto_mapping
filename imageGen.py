import PIL.Image
import PIL.ImageDraw
from parseCSV import get_polygons_from_csv

def generate_image(output, size, polygons): # polygons is a list of polygons where each polygon is a list of (x, y)-tuples
    width, height = size, size
    im = PIL.Image.new(mode='RGB', size=(size, size))
    draw = PIL.ImageDraw.Draw(im)
    for i, corners in enumerate(polygons):
        draw.polygon(corners, fill="white")
    im.save(output, format='tiff')

def image_from_csv():
    polygons = get_polygons_from_csv("data/csv/data_eesti.csv")
    generate_image("data/img/tartu.tif", 25000, polygons)

image_from_csv()
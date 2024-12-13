import PIL.Image
import PIL.ImageDraw
import os
from parseCSV import get_polygons_from_csv
from itertools import product
from random import uniform

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


def generate_train_validation_set(train_path, validation_path, input_image, target_image, data_path, validation_weight=0.2, tile_size=250):
    PIL.Image.MAX_IMAGE_PIXELS = None
    input_name, input_ext = os.path.splitext(input_image)
    input_img = PIL.Image.open(os.path.join(data_path, input_image))

    target_name, input_ext = os.path.splitext(target_image)
    target_img = PIL.Image.open(os.path.join(data_path, target_image))
    
    w, h = input_img.size

    grid = product(range(0, h, tile_size), range(0, w, tile_size))
    for i, j in grid:
        r = uniform(0, 1)
        output_path=train_path
        name='train'
        if r <= validation_weight:
            output_path = validation_path
            name = 'validation'
        box = (j, i, j+tile_size, i+tile_size)
        out_input = os.path.join(os.path.join(output_path, 'input'), f'{name}_{int(i*w/tile_size/tile_size+j/tile_size)}{input_ext}')
        out_target = os.path.join(os.path.join(output_path, 'target'), f'{name}_{int(i*w/tile_size/tile_size+j/tile_size)}{input_ext}')
        input_img.crop(box).save(out_input)
        target_img.crop(box).save(out_target)

polygons = get_polygons_from_csv('../data/csv/data_eesti.csv')
generate_image('data/img/target.tif', 25000, polygons)
generate_train_validation_set('data/train', 'data/validation', 'input.tif', 'target.tif', 'data/img')
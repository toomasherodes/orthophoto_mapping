# OTRHOPHOTO MAPPING (D12)

### Introduction
 The goal of the project is to identify all man-made buildings from orthophotos with machine learning. Ideally also find all buildings that should be registered but are not.


## Methology

The model is trained on a DeepLabv3 model using our custom dataset consisting of orthophotos from Tartu. It uses atrous convolution for semantic image segmentation [(reference)](https://arxiv.org/pdf/1706.05587) - this is perfect for our defined criteria.  We tried different weights for the segment classes and different learning rates. We also tried training it for different amounts of epochs - we concluded that it started overfitting very quickly, so we had to train it less for now. The input data is a 250x250px color-infrared orthophoto and the output is a segmented bitmap of where the model thinks the buildings lie. Using infrared photos makes it easier for the model to recognize man-made structures

## Results

The trained model was capable of identifying buildings on an orthophoto. However, the detection accuracy was low and not all buildings were correctly identified, smaller houses mostly went undetected. Moreover, the buildings were never fully detected and the model was only capable of drawing a blob in the middle of the house.

[model input](data/img/sample/model_input.png)
[model output](data/img/sample/model_output.png)
[truth](data/img/sample/truth.png)

## Data sources

Orthophotos can be downloaded [here](https://geoportaal.maaamet.ee/est/ruumiandmed/ortofotod-p99.html)

Data about buildings can be requested from [here](https://livekluster.ehr.ee/ui/ehr/v1/infoportal/reports)

Coordinates are in wkb format and need to be converted, [shapely](https://pypi.org/project/shapely/) works well.

Coordinate data also needs to be filtered by location, building type and date. All the scripts are in the src folder.

There is also some raw sample data in the data folder.

## Model Training guide    
### largerly followed [this tutorial to train the model](https://debuggercafe.com/train-pytorch-deeplabv3-on-custom-dataset/)

1. aquire training input and target image, place them in `/data/img` with filenames `input.tif` and `target.tif` respectively
2. to create train and validation data, run `generate_train_validation_set('./data/train', './data/validation', 'input.tif', 'target.tif', './data/img')`
3. to start training the model, run `python train.py --epochs 30 --batch 16` from the `/src` folder
#### todo: fix train.py

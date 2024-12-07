### largerly followed [this tutorial to train the model](https://debuggercafe.com/train-pytorch-deeplabv3-on-custom-dataset/)

1. aquire training input and target image, place them in `/data/img` with filenames `input.tif` and `target.tif` respectively
2. to create train and validation data, run `generate_train_validation_set('./data/train', './data/validation', 'input.tif', 'target.tif', './data/img')`
3. to start training the model, run `python train.py --epochs 30 --batch 16` from the `/src` folder
#### todo: fix train.py

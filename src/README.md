## Files in this folder

#### config.py 
Config file for the images created

#### datasets.py
Dataset preparation scripts

#### engine.py
Functions for training and validation (also has a custom loss function that was ultimately not used)

#### imageGen.py
Functions to create test/val/train images for the model

#### koordinaadidTranslate.py
Unused function for mapping between pixel and real-life coordinates

#### metrics.py
Pixel accuracy calculation function

#### parseCSV.py
Functions/scripts for exstracting data from csv file (see sample in data/csv).

#### test.py
Tool for testing the model

#### train.py
Model training (run this to train the model, use custom parameters `--epochs`, `--batch` and `--lr` to tweak the number of epochs, batches and the learning rate)

#### utils.py
Tools for displaying, overlaying, analyzing etc images.
# SERT

Here is the open-source code for *SERT*. Below is a detailed introduction to this repository:

## Folders

1. The ***./satellite_encoder*** folder contains the code for extracting satellite imagery features, which corresponds to the implementation of the *image encoder* described in the paper. The ***simclr*** folder contains the implementation code for the image encoder model.

2. The ***./sert_framework*** folder contains the core code for SERT, which includes the overall transfer learning model architecture. ***model.py*** is the implementation code for the transfer learning framework.

## Dataset

The ***./sert_framework/data*** folder contains the flow datasets required for training SERT. The dataset includes data from three cities, with each city containing flow data for two types of transportation: Taxi and Bike.
Please extract the compressed file before use.

## Running the code

You need to first run the code in the ***./satellite_encoder*** folder to extract auxiliary features for the subsequent matching computations.  

1. Run `positive_pair_poi.py` and `positive_pair_geo.py` to construct positive and negative samples for contrastive learning.

2. Run `main.py` to train the image feature extraction model.

3. Run `feature_extract.py` to extract the image features.

Afterward, you can run `train.py` in the ***./sert_framework*** folder to train the entire transfer learning model and obtain the prediction results.

## Version Information

The entire SERT's source code is implemented using **Python version 3.7.16** and **torch version 1.8.1+cu111**.


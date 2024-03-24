# A deep learning method for cultivated land parcels (CLPs) delineation with high-generalization capability
The official PyTorch implementation of "A deep learning method for cultivated land parcels (CLPs) delineation with high-generalization capability".


## Sample process
If your images and samples are unsegmented, then put the entire image and samples in the:  
`<ImageDataset/data>`  
and   
`python sample_process.py`   
**python sample_process.py** can segment the whole image and boundary/region samples into sample tiles 
and calculate the distance samples  
Note:Please check the path **python sample_process.py**.  


## Requirements
`PyTorch  
TensorboardX  
GDAL  
OpenCV  
PIL  
numpy  
tqdm  
scikit-learn`  <br>

## run the model
The code is tested under a Linux desktop with torch 1.2.0 and Python 3.6 <br>
Train model:<br>
`python train.py`<br>
prediction:<br>
`python predict_boundary.py`<br>
`python predict_region.py`<br>

## Pretrained weight

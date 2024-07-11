### Computer Vision Algorithm for the Object Detection of Polygons


Here a folder that stores the data and hopefully does automatic training/testing of the model performance for the lunar zebro "traici" 2024 conference will be created.

The objectives of the repository: 

- 1. Have files to allow training and testing of the model
- 2. Have an easy way to download/access the data
- 3. Create Github Actions in order to automatically train/test the model based on the desired configurations
- 4. Create Combine with google bucket so that github actions performs training automatically with key



## How to use the repository 

If you want to train a model locally, please download this repository and the click the following link to download the data:\

[google bucket](https://storage.googleapis.com/polygon_bucket/data.zip)



## Converting the model to Onnx


In order to convert the model to ONNX format (which will later be used on the rover), please follow the following steps (assumed on linux terminal):

```
yolo export model=*insert your model* format=onnx opset=13 simplify
```

in my case this was
```
yolo export model=weights/best_nano10.pt format=onnx opset=13 simplify
```


## Model Testing

Currently the model inference using the ONNX on real data is bad. For this reason the model was not succesfully loaded into the final rover code. Nonetheless, the preprocessing steps are performed and tested within the ```test_onnx.py``` file in the main directory. 

In later stages, the idea is that the same preprocessing will be done on the final rover.
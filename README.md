# Facial keypoint detection
Using TensorFlow 

***V2 rework in progress!***

## Example
|           My implementation:           |           DLib reference:           |
| :------------------------------------: | :---------------------------------: |
| ![My implementation](static/mine.jpeg) | ![DLib reference](static/dlib.jpeg) |

## Usage:
- git clone
- pip install -r requirements.txt
- python3 main.py
- send POST request to /facial-landmark-detection with 'image' parameter
- interactive web verison availible at /
- docker container availible at [dockerhub](https://hub.docker.com/repository/docker/empyempt/fld)  


~~~bash
docker pull empyempt/fld:latest
~~~

## Dataset: 
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)  

Exact images, compressed images and .csv files can be pulled via [DVC](https://dvc.org/)
~~~bash
dvc pull
~~~

## Model  
Custom CNN implemented in Keras, exact architecture can be found in model.py


## Landmarks:  
-   Jaw Points = 0–16
-   Right Brow Points = 17–21
-   Left Brow Points = 22–26
-   Nose Points = 27–35
-   Right Eye Points = 36–41
-   Left Eye Points = 42–47
-   Mouth Points = 48–60
-   Lips Points = 61–67


## Training and tinkering

You can run predictions, train new models on existing dataset and create new datasets interactively in provided Notebook (example.ipynb)


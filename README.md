# Facial-landmark-detection

## Usage:
- git clone
- pip install -r requirements.txt
- python3 main.py
- send POST request to /facial-landmark-detection with 'image' parameter
- interactive web verison availible at /
- docker container availible at [dockerhub](https://hub.docker.com/repository/docker/empyempt/fld)
~~~bash
docker pull empyempt/fld:1.0    
~~~

## Datasets: 
[Male and female faces dataset Kaggle](https://www.kaggle.com/ashwingupta3012/male-and-female-faces-dataset/metadata)  
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

Exact images and .csv files can be pulled via dvc
~~~bash
dvc pull
~~~

## Model
CNN implemented in Keras (exact architecture can be found in model.py)

## Landmarks:  
-   Jaw Points = 0–16
-   Right Brow Points = 17–21
-   Left Brow Points = 22–26
-   Nose Points = 27–35
-   Right Eye Points = 36–41
-   Left Eye Points = 42–47
-   Mouth Points = 48–60
-   Lips Points = 61–67


## Output
Model outputs tensor with shape (136, ) where for (k in 0->67) 
- [k] is X for landmark k
- [k+1] is Y for landmark k

## Training and tinkering

You can train model yourself on your own dataset  
__It's best to use config files here__  

First you need to prepare a csv file, which can be done by 
```python
import data.code.images_to_csv as prep
prep.images_to_csv('dataset_path', 'output_path')
```

Now you can train the model
```python
import train as tr
# most of this parameters are taken from config.py as defaults
model = tr.train_new_model(labels = 'path_to_csv', epochs = 30, checkpoints = False)
```

Finally, you can see your model working
```python
import run
#to predict on image
run.predict_on_image(model, image_path  = 'image_path', output = 'output_path')
# to predict on video stream from webcam (press esc to stop)
run.predict_stream(model)
```

# Face keypoint detection

Using TensorFlow

## Example

|           My implementation:           |           DLib reference:           |
| :------------------------------------: | :---------------------------------: |
| ![My implementation](static/mine.jpeg) | ![DLib reference](static/dlib.jpeg) |

## Usage

- git clone
- pip install -r requirements.txt
- python3 main.py
- send POST request to /fkd with 'image' parameter
- interactive web verison availible at /

Alternatively you can use it via Docker container

- [V1 dockerhub](https://hub.docker.com/repository/docker/empyempt/fld)
- [V2 dockerhub](https://hub.docker.com/repository/docker/empyempt/fkd)

```bash
docker pull empyempt/fkd:latest
```

## V1 & V2 differences  

My goal with V2 was to make it run faster and have more accurate predictions  

V2 does `not use` any face detection algorithms while V1 does  
This is both a curse and a blessing:  
V2 is much faster due to code optimizations and less preprocessing time  
At the same time V2 is less versatile bacause of this: it expects a photo where the face is somewhat localized  
I wanted to train the model with augmented images, but I didn't, I'm too burned out of this project due to various issues

## Dataset

[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

~~Exact images, compressed images and .csv files can be pulled via [DVC](https://dvc.org/)~~

~~```bash~~
~~dvc pull~~
~~```~~

Not anymore... I took it down, might reupload later  

## Model

Custom CNN implemented in Keras, exact architecture can be found in model.py

## Landmarks

- Jaw Points = 0–16
- Right Brow Points = 17–21
- Left Brow Points = 22–26
- Nose Points = 27–35
- Right Eye Points = 36–41
- Left Eye Points = 42–47
- Mouth Points = 48–60
- Lips Points = 61–67

## Model output

Model outputs a tensor of shape `[batch_size, 68, 2]`, which corresponds to 68 xy coordinates in `[0...1]` range

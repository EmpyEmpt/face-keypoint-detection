# Datasets: 
https://www.kaggle.com/ashwingupta3012/male-and-female-faces-dataset/metadata
https://github.com/NVlabs/ffhq-dataset (https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)
# Landmarks:  
-   Jaw Points = 0–16
-   Right Brow Points = 17–21
-   Left Brow Points = 22–26
-   Nose Points = 27–35
-   Right Eye Points = 36–41
-   Left Eye Points = 42–47
-   Mouth Points = 48–60
-   Lips Points = 61–67

# Explanation
Processed dataset consists of:
TLx, TLy, BRx, BRy, x0, y0, x1, y1, ..., x67, y67
(x, y) are relative to the upper left corner of bounding box

    ex. = image with size (300, 300)
    TL, BR points to area from (100, 100) to (200, 200)
    x0, y0 = 50, 50 would correspond to 150, 150 on original image 

# Output
Model outputs tensor with shape (136, ) where for (k in 0->67) 
[k] is X for landmark k
[k+1] is Y for landmark k
   
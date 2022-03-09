# Datasets: 
https://www.kaggle.com/ashwingupta3012/male-and-female-faces-dataset/metadata
# Rects:  
-       [0] - Upper left corner of face
-       Left[1] - Lower right corner of face
# Landmarks:  
-       Jaw Points = 0–16
-       Right Brow Points = 17–21
-       Left Brow Points = 22–26
-       Nose Points = 27–35
-       Right Eye Points = 36–41
-       Left Eye Points = 42–47
-       Mouth Points = 48–60
-       Lips Points = 61–67


1. Explanation
   Dataset is 
   [[TLx, TLy], [BRx, BRy]], [x0, y0], [x1, y1], ..., [x67, y67]
    (x, y) are relative to the Bounding Box (First pair (of pairs))

    ex. = image with size (300, 300)
    TL, BR points to arean from (100, 100) to (200, 200) 

2. Transformation
   1. Calc width, height of BB
   2. Transfrom (x, y) ex. (50, 40) -> (0.5, 0.4)
        so that 0.5 * width = x & 0.5 * height = y
        { Such coords. will be named Realative (Rx, Ry) }

3. FEED THE MONSTER
   1. Crop out face(s)
   2. Resize crops to smth (idk, 128x128) and accordingly resize (Rx, Ry)
   3. FEED IT

4. Output
   As output we get 68 pairs of (Rx, Ry) 
   1. Resize (Rx, Ry) backwards, accordingly to step 3.2 to get (x, y)
   2. Resize (x, y) backwards, accordingly to step 2.2
   3. Plot them accordingly to BB (or ffs transform backwards to be relative to original image an plot that way)
   
5. PROFIT!
   


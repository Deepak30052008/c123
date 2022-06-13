import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X,y = fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xtrain_scaled=xtrain/255.0
xtest_scaled=xtest/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrain_scaled,ytrain)
ypred=clf.predict(xtest_scaled)
import cv2
capture=cv2.VideoCapture(0)
while (True):
    ret,frame=capture.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height,width=grey.shape
    upperLeft=(int(width/2-56),int(height/2-56))
    bottomRightCorner=(int(width/2+56),int(height/2+56))
    cv2.rectangle(grey,upperLeft,bottomRightCorner,(0,255,0),2)
    roi=grey[upperLeft[1]:bottomRightCorner[1],upperLeft[0]:bottomRightCorner[0]]
    imageConverted=Image.fromarray(roi)
    image1=imageConverted.convert('L')
    imageResized=image1.resize((28,28),Image.ANTIALIAS)
    imageResizeInverted=PIL.ImageOps.invert(imageResized)
    pixelFilter=20
    minimumPixel=np.percentile(imageResizeInverted,pixelFilter)
    imageResizeInvertedScaled=np.clip(imageResizeInverted-minimumPixel,0,255)
    maximumPixel=np.max(imageResizeInverted)
    imageResizeInvertedScaled=np.asarray(imageResizeInvertedScaled)/maximumPixel
    testSample=np.array(imageResizeInvertedScaled).reshape(1,784)
    testPredict=clf.predict(testSample)
    print(testPredict)
    cv2.imshow("output",grey)
    if(cv2.waitKey(1)&0xFF==ord('q')):
        break

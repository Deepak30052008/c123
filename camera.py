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
    cv2.imshow("output",grey)
    if(cv2.waitKey(1)&0xFF==ord('q')):
        break

import cv2 as cv
import numpy as np

#initialise detectors
face_detector=cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_detector=cv.CascadeClassifier('cascades/haarcascade_eye.xml')

#if eyes are undetectable, return cropped face
def face(image,gray):
    faces=face_detector.detectMultiScale(gray,1.1,7)
    cropped_image=image.copy()
    for (x,y,w,h) in faces:
        print('Face detected at - ',x,y,w,h)
        cropped_image=image[y:y+h,x:x+h]
    return cropped_image

def align(image):

    #resize image if necessary
    cols,rows,_=image.shape
    if(cols>800):
        rows=int((800/rows)*cols)
        cols=800
        image=cv.resize(image,(cols,rows))

    #convert image to grayscale for detection
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    #detect faces, create faces as region of interest(ROI) and find eyes inside ROI
    eyes=eye_detector.detectMultiScale(gray)
    detect_image=image.copy()
    faces=face_detector.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in faces:
        roi=gray[y:y+h,x:x+w]
        eyes=eye_detector.detectMultiScale(roi,1.3,10)
    #if number of eyes found, are less than two, return the default resized image
    try:
        leftEye,rightEye=eyes[0],eyes[1]
    except IndexError:
        print('Eyes undetectable!')
        return face(image,gray)

    #calculate center of rotation and angle of rotation
    #centerX and centerY is center of rotation
    centerX=int(min(leftEye[0]+leftEye[2]/2,rightEye[0]+rightEye[2]/2)+abs((leftEye[0]+(leftEye[2]/2))-(rightEye[0]+(rightEye[2]/2)))/2)
    centerY=int(min(leftEye[1]+leftEye[3]/2,rightEye[1]+rightEye[3]/2)+abs((leftEye[1]+(leftEye[3]/2))-(rightEye[1]+(rightEye[3]/2)))/2)

    #angle is negative of, slope of lines joining, center of left and right eye-boxes
    angle=np.arctan((leftEye[1]+(leftEye[3]/2)-centerY)/(leftEye[0]+(leftEye[2]/2)-centerX))*(180/3.1412)

    #calculate transformation matrix
    M=cv.getRotationMatrix2D((centerX,centerY),angle,1)

    #detect face and rotate image
    faces=face_detector.detectMultiScale(gray,1.1,5)
    cols,rows,_=image.shape
    image=cv.warpAffine(image,M,(rows,cols),flags=cv.INTER_LINEAR)

    #crop face area
    cropped_image=image.copy()
    for (x,y,w,h) in faces:
        print('Second Face detected at - ',x,y,w,h)
        cropped_image=image[y:y+h,x:x+h]

    #return final image
    return cropped_image

#driver code
if __name__=='__main__':
    #replace with full image path. Example - 'D:/img/image.jpg'
    image=cv.imread('Image Path',1)
    cropped_image=align(image)
    cv.imshow('Final Image',cropped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

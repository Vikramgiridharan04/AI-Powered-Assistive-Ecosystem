import cv2
import face_recognition
import os

name=input('Enter Your Name: ')
try:
    if name.isalnum() or name.isalpha():
        cam=cv2.VideoCapture(0)
        harcascadePath='haarcascade_frontalface_default.xml'
        detector=cv2.CascadeClassifier(harcascadePath)
        while True:
            ret,img=cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=detector.detectMultiScale(img,1.3,5)
            cv2.imshow('Frame',img)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                if cv2.waitKey(100) & 0xFF==ord('c'):
                    cv2.imwrite('face_images/'+name+'.jpg',gray[y:y+h,x:x+w])    
                    break
            if cv2.waitKey(100) & 0xFF==ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
        res=name.strip()
        print(res)
    else:
        print("Enter Your Name Correctly")
finally:
    print('hi')
                

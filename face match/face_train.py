import cv2


import os
name=input("Enter Your Name :")
try:
    if name.isalnum() or name.isalpha():
        cam=cv2.VideoCapture(0)
        harcascadePath="haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            cv2.imshow('frame',img)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                if cv2.waitKey(100) & 0xFF == ord('c'):# Press c button to capture your face
                    cv2.imwrite("face_images/"+name+ ".jpg", gray[y:y+h,x:x+w])
                    break
            if cv2.waitKey(100) & 0xFF == ord('q'):# press q button to close the camera portal
                break
            
        cam.release()
        cv2.destroyAllWindows()
        res='Name: '+ name.strip()
        print(res)
    else:
            print("Put the Correct Name")
except:
    pass

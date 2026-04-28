import cv2
import os

# Create directory for storing images if it doesn't exist
if not os.path.exists("face_images"):
    os.makedirs("face_images")

name = input("Enter Your Name: ")

try:
    if name.isalnum() or name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        captured = False

        while not captured:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('frame', img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                for (x, y, w, h) in faces:
                    cv2.imwrite(f"face_images/{name}.jpg", gray[y:y+h, x:x+w])
                    captured = True
                    print(f"Image captured for {name}")
                    break  # Exit for loop after capturing

            elif key == ord('q'):
                print("Exiting without capturing image.")
                break

        cam.release()
        cv2.destroyAllWindows()

        if captured:
            res = 'Name: ' + name.strip()
            print(res)
    else:
        print("Please enter a valid name (alphanumeric or alphabetic characters only).")
except Exception as e:
    print(f"An error occurred: {e}")

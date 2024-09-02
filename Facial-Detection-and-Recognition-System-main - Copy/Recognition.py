# import cv2
# import numpy as np

# # Load the saved image and convert it to grayscale
# Shan_image = cv2.imread("Shan.jpg")
# Shan_gray = cv2.cvtColor(Shan_image, cv2.COLOR_BGR2GRAY)

# # Initialize the face recognizer
# face_cascade = cv2.CascadeClassifier("D:\python\Facial-Detection-and-Recognition-System-main\haarcascade_frontalface_default (3).xml")
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Detect the face in the saved image
# Shan_faces = face_cascade.detectMultiScale(Shan_gray, 1.1, 4)
# for (x, y, w, h) in Shan_faces:
#     # Train the recognizer on the detected face
#     recognizer.train([Shan_gray[y:y+h, x:x+w]], np.array([1]))

# # Start the webcam to recognize your face in real-time
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# while True:
#     success, img = cap.read()
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    
#     for (x, y, w, h) in faces:
#         face = img_gray[y:y+h, x:x+w]
#         id, confidence = recognizer.predict(face)
        
#         # Check if the recognized face matches the saved face
#         if id == 1 and confidence < 50:  
#             name = "Shan"
#         else:
#             name = "Unknown"
        
#         # Display the name and rectangle around the face
#         cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#     cv2.imshow("Face Recognition", img)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





import cv2
import numpy as np

# Load the saved image and convert it to grayscale
Shan_image = cv2.imread(r"D:\python\Facial-Detection-and-Recognition-System-main\Shan.jpg")

# Check if image was loaded correctly
if Shan_image is None:
    raise FileNotFoundError("The image file 'Shan.jpg' was not found. Please check the file path.")

Shan_gray = cv2.cvtColor(Shan_image, cv2.COLOR_BGR2GRAY)

# Initialize the face recognizer
face_cascade = cv2.CascadeClassifier(r"D:\python\Facial-Detection-and-Recognition-System-main\haarcascade_frontalface_default (3).xml")

# Check if cascade file was loaded correctly
if face_cascade.empty():
    raise FileNotFoundError("The Haar cascade file was not found. Please check the file path.")

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Detect the face in the saved image
Shan_faces = face_cascade.detectMultiScale(Shan_gray, 1.1, 4)
for (x, y, w, h) in Shan_faces:
    # Train the recognizer on the detected face
    recognizer.train([Shan_gray[y:y+h, x:x+w]], np.array([1]))

# Start the webcam to recognize your face in real-time
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = img_gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face)
        
        # Check if the recognized face matches the saved face
        if id == 1 and confidence < 50:  
            name = "Shan"
        else:
            name = "Unknown"
        
        # Display the name and rectangle around the face
        cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


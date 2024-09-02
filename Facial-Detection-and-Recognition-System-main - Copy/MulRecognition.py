import cv2
import numpy as np

# Initialize the face recognizer and face cascade
face_cascade = cv2.CascadeClassifier(r"D:\python\Facial-Detection-and-Recognition-System-main\haarcascade_frontalface_default (3).xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Training images and labels
image_paths = [r"D:\python\Facial-Detection-and-Recognition-System-main\Kunal.jpg", r"D:\python\Facial-Detection-and-Recognition-System-main\Shan.jpg"]
labels = []
training_images = []

# Process each image for training
for idx, image_path in enumerate(image_paths):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at path '{image_path}' not found.")
        continue
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in faces:
        training_images.append(img_gray[y:y+h, x:x+w])
        labels.append(idx + 1)  # Label: 1 for Kunal, 2 for Shan

# Convert labels to numpy array
labels = np.array(labels)

# Train the recognizer
if len(training_images) > 0:
    recognizer.train(training_images, labels)
    recognizer.save('trainer.yml')
    print("Training completed and model saved.")
else:
    print("No faces found in the training images. Training not performed.")

# Start the webcam to recognize faces in real-time
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
        
        # Check the recognized face and assign name
        if confidence < 50:  # Confidence threshold
            if id == 1:
                name = "Kunal"
            elif id == 2:
                name = "Shan"
            else:
                name = "Unknown"
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

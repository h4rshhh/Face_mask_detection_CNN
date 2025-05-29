import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Loading the trained model
model = load_model('model_full.keras', compile=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (128, 128))
        face_array = image.img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = face_array / 255.0  # normalize if your model was trained with normalized input

        prediction = model.predict(face_array)

        label = "Mask" if prediction[0][0] < 0.5 else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Real-Time Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from keras.models import load_model
import time

def main():
    try:
        # Load the trained model
        model = load_model('./emotion_model.h5')
    except IOError:
        print("Error: Failed to load the model. Check the model path.")
        return

    # Load Haar Cascade for face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print(f"Error: {e}")
        return

    # Emotion label mapping
    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    # Open a video stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Frame rate control
    frame_rate = 10  # You can adjust this value
    prev = 0

    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            break

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                # Resize and normalize the ROI
                resized_face = cv2.resize(face_roi, (48, 48)) / 255.0
                reshaped_face = resized_face.reshape(1, 48, 48, 1)

                # Predict the emotion
                emotion_prediction = model.predict(reshaped_face)
                emotion_label = np.argmax(emotion_prediction)

                # Display the emotion text
                cv2.putText(frame, emotion_labels[emotion_label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Resize the frame for smaller window
            frame_resized = cv2.resize(frame, (640, 480))  # You can adjust the size

            # Display the resulting frame with emotion
            cv2.imshow('Emotion Detection', frame_resized)

            # Break the loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

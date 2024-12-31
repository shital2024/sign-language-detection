import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('sign_language_model.h5')

# Define the labels for the gestures
labels = ['A', 'B', 'C', 'D', 'E']  # Replace with your actual labels

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """
    Preprocess the frame to match the input shape of the model.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to the model's expected input size
    resized = cv2.resize(gray, (64, 64))  # Adjust size to match your model
    # Normalize pixel values
    normalized = resized / 255.0
    # Expand dimensions to match model input
    return np.expand_dims(normalized, axis=(0, -1))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Define region of interest (ROI) for hand detection
    x_start, y_start, x_end, y_end = 100, 100, 300, 300
    roi = frame[y_start:y_end, x_start:x_end]

    # Preprocess the ROI
    processed_roi = preprocess_frame(roi)

    # Make a prediction
    prediction = model.predict(processed_roi)
    predicted_label = labels[np.argmax(prediction)]

    # Display prediction on the frame
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    cv2.putText(frame, predicted_label, (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

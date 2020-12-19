import cv2
import tensorflow as tf

# Open a handle to the default webcam
camera = cv2.VideoCapture(0)


# Start the capture loop
while True:
    # Get a frame
    ret_val, frame = camera.read()
    frame = cv2.resize(frame, (418, 216))

    # Show the frame
    cv2.imshow('Webcam Video Feed', frame)
    new_model = tf.keras.models.load_model('model.h5')
    new_model.predict(frame)

    # Stop the capture by hitting the 'esc' key
    if cv2.waitKey(1) == 27:
        break

# Dispose of all open windows
cv2.destroyAllWindows()
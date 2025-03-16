import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import mediapipe as mp

# Load your trained model
model = load_model('./model/my_cnn_model.h5')

# Define your class names as they appear in your dataset (order matters!)
class_names = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'Ө', 'П', 'Р', 'С', 'Т', 'У', 'Ү', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# Set the expected input size (should match the training size)
IMG_WIDTH, IMG_HEIGHT = 300, 225

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make a copy of the frame for cropping the hand ROI without annotations
    original_frame = frame.copy()

    # Convert frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_roi = None

    # Check if a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate bounding box coordinates based on hand landmarks
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_coords) * w)
            xmax = int(max(x_coords) * w)
            ymin = int(min(y_coords) * h)
            ymax = int(max(y_coords) * h)

            # Add a margin around the hand region
            margin = 20
            xmin = max(xmin - margin, 0)
            ymin = max(ymin - margin, 0)
            xmax = min(xmax + margin, w)
            ymax = min(ymax + margin, h)

            # Crop the hand region from the original frame (without drawn annotations)
            hand_roi = original_frame[ymin:ymax, xmin:xmax]

            # Process only the first detected hand
            break

    # If a hand ROI is detected, process it and display only the input image to the model
    if hand_roi is not None and hand_roi.size != 0:
        # Convert the ROI from BGR to RGB and resize it
        hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(hand_rgb)
        im = im.resize((IMG_WIDTH, IMG_HEIGHT))
        input_img = np.array(im)  # This image is in RGB

        # Prepare image for display (convert to BGR for OpenCV)
        input_img_disp = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

        # Expand dimensions and get prediction from the model
        img_array = np.expand_dims(input_img, axis=0)
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_index]

        print(predicted_label)

        # Overlay the predicted label on the input image for display
        cv2.putText(input_img_disp, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show only the processed input image
        cv2.imshow("Input Image", input_img_disp)
    else:
        # If no hand is detected, display a blank image of the same size
        blank_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        cv2.imshow("Input Image", blank_img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

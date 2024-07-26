import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import time

def extract_data(image):
    frame = cv2.resize(image, (640, 480))  # Smaller size for faster processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(rgb_frame)
    
    result = [0] * 63  # create empty list with 21*3 coordinates
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark  # We only process one hand
        for i, landmark in enumerate(landmarks):
            idx = i * 3
            result[idx] = landmark.x
            result[idx + 1] = landmark.y
            result[idx + 2] = landmark.z
    
    return result

def on_input(label):
    print(label)
    print(type(label))
    match label:
        case 1:  # Pushing Hand Away
            pyautogui.click()
        case 2: # Sliding Two Fingers Down
            pyautogui.move(0,20,0)
        case 5: # Sliding Two Fingers Up
            pyautogui.move(0,-20,0)
        case 3: # Sliding Two Fingers Left
            pyautogui.move(-20,0,0)
        case 4: # Sliding Two Fingers Right
            pyautogui.move(20,0,0)
        case 8: # Zooming In With Two Fingers
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(100)
            pyautogui.keyUp('ctrl')
        case 9: # Zooming Out With Two Fingers
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(-100)
            pyautogui.keyUp('ctrl')
        case default:
            print("Nothing happened")

# Define the RNNClassifier class
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

if __name__=="__main__":
    # Use MediaPipe to draw the hand framework over the top of hands it identifies
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,  # Use the fastest model
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8)

    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Model hyperparameters
    input_size = 64
    hidden_size = 256
    num_layers = 2
    num_classes = 10
    dropout = 0.5

    # Initialize and load the model
    model = RNNClassifier(input_size, hidden_size, num_layers, num_classes, dropout)
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(device)
    model.load_state_dict(torch.load('final_mit0.pth', map_location=device))
    model.to(device)
    model.eval()

    # dict containing labels with the corresponding label number
    label_dict = {0: "Doing other things",
                1: "Pushing hands away",
                2: "Sliding two fingers down",
                3: "Sliding two fingers left",
                4: "Sliding two fingers right",
                5: "Sliding two fingers up",
                6: "Thumbs down",
                7: "Thumbs up",
                8: "Zooming in with two fingers",
                9: "Zooming out with two fingers"}


    row_arr = []            # Stores the coordinates of the previous frames
    n = 16                  # Number of frames for prediction
    last_prediction = []    # Stores the labels of the previous frames

    row_arr = []
    counter, ct = 0, 0
    frame_counter = 0

    while cap.isOpened():
        frame_counter += 1
        counter += 1
        frame_start = time.time()

        ret, frame = cap.read()
        row_arr.extend(extract_data(frame))

        if counter > n:
            del row_arr[0:63]
            counter -= 1
        if counter == n: # 0.005
            # reshape row_arr
            x = np.concatenate([np.array(row_arr).reshape(16, 63), np.zeros((16, 1))], axis=1)[None, :, :]
            input_data = torch.tensor(x, dtype=torch.float32).to(device)
            
            if np.sum(x == 0) / x.size < 0.5:  # Only predict if enough data points are available (>50%)
                with torch.no_grad():  # Inference
                    output = model(input_data)[0]
                    label = torch.argmax(output).cpu().item()
                
                cv2.putText(frame, str(label_dict[label]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, thickness=5)
                last_prediction = str(label_dict[label])
            
            cv2.imshow("Frame", frame) # 0.005
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if frame_counter %10 == 0:
                print(f"FPS: {1.0 / (time.time() - frame_start):.2f}")
                print(last_prediction)
                print()

            # # uncomment following lines to get mouse movement
            # if label and (label == 2 or label == 3 or label == 4 or label == 5):
            #     on_input(label)
            # elif label and (label == 1 or label == 8 or label == 9):
            #     if frame_counter%10 == 0:
            #         on_input(label)

    # Release the camera and file writer
    cap.release()
    cv2.destroyAllWindows()

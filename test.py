import cv2
from ultralytics import YOLO
import numpy
from termcolor import colored, cprint
import time
import os
# Load the YOLOv8 model
model = YOLO('best.pt')
cap = cv2.VideoCapture('/workspaces/BJ-card-counter/demo.mp4')
ognames = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']
names = ['AH', '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', '10H', 'JH', 'QH', 'KH', 'AD', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', 'JD', 'QD', 'KD', 'AC', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', '10C', 'JC', 'QC', 'KC', 'AS', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', '10S', 'JS', 'QS', 'KS']
frame_checker = 0 
onscreen = 0 
count = 0
check = None
def print_cards():
    for i in names[:13]:
        print(colored(i,"red"), end=' ')
    print()
    for i in names[13:26]:
        print(colored(i,"red"), end=' ')
    print()
    for i in names[26:39]:
        print(colored(i,"red"), end=' ')
    print()
    for i in names[39:52]:
        print(colored(i,"red"), end=' ')
    print()
        
print_cards()
detected = []
while cap.isOpened():
        # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)
        ids = results[0].boxes.cls.cpu().numpy().astype(int)
        print(ids)
        dids = list(set(ids))
        print(dids)
        if dids != check:
            for i in dids:
                detected.append(ognames[i])
                detected = list(set(detected))
            for l in detected:
                if l in names:
                    target = names.index(l)
                    names[target] = colored(l, "green")
            os.system("clear")
            print_cards()
        else:
            os.system("clear")
            print_cards()
            
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if detected != []:
            frame_checker += 1
        if frame_checker == 2 and detected != []:
            frame_checker = 0
            onscreen = 1
        if onscreen == 1 and dids != check:
            for x in dids:
                if 5 < int(x) < 24:
                    count += 1
                elif 25 < int(x) < 36:
                    count = count 
                else:
                    count -= 1
        detected = []
        onscreen = 0
        check = dids
        names = ['AH', '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', '10H', 'JH', 'QH', 'KH', 'AD', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', 'JD', 'QD', 'KD', 'AC', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', '10C', 'JC', 'QC', 'KC', 'AS', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', '10S', 'JS', 'QS', 'KS']
        # Break the loop if 'q' is pressed
        print(count)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

    


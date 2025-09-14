from ultralytics import YOLO
import cv2
import numpy as np

if __name__ == '__main__':

    model = YOLO("best.pt")
    results = model.train(data="/Users/orierez/Desktop/Blackjack/data/data.yaml", epochs=100, device='mps') 

    #^ Runs the training algorithim
    

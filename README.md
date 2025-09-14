This program uses Yolov8 in order to count cards in blackjack. There are two versions, one that uses the webcam and one that captures the screen.

At the moment, the screen capture one isn't completely accurate, as I haven't been able to find a large enough dataset of video blackjack in order to train it properly. The webcam one is a lot more accurate.
You can use the train.py file in order to train the dataset more, however you should probably add your own dataset and replace the pt file, as this model is basically completely trained on the current (relatively small) dataset.


The screen grab version captures the upper left portion of your screen. This can easily be changed in the program by changing the bounding box. 

DISCLAIMER:
I am not responsible for any money lost while using this, it was created simply to expand my knowledge on computer vision and practice using it.

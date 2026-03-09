import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict, Counter


class JerseyReader:
    def __init__(self):
        # Initialize the OCR model only for numbers, language eng
        self.ocr = PaddleOCR(
            use_angle_cls = False, # jerseys are usually horizontal, no need for angle classification
            use_gpu = False, # set to True if you have a compatible GPU for faster processing
            lang = "en",
            show_log = False,
        )

        self.readings = defaultdict(list) # store OCR readings for each player ID to improve accuracy with multiple readings
        self.min_readings = 5 # minimum number of readings required to determine a jersey number
        self.confirmed =  {} # store confirmed jersey numbers for each player ID


    def extract_jersey_crop(self, frame, bbox, crop_ratio = 0.5):
        # extract a crop of the jersey area from the bounding box, we will focus on the upper part of the bounding box where the jersey number is usually located
        
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
        # calculate height
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        # get only upper section of the bounding box
        crop_y2 = y1 + int(height*crop_ratio)
        crop = frame[y1:crop_y2, x1:x2] # actual crop of the image
        if crop.size == 0:
            return None
        return crop
    

    def preprocess_crop(self, crop):
        # improve crop to facilitate OCR readings
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # conversion to a scaleof greys
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        binary = cv2.adaptiveThreshold(
            gray, 255
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        return binary
    

    def read_number(self, crop):
        # perform OCR on the crop and returns found number or Non if no number was found
        result = self.ocr.ocr(crop, cls=False)

        if not result or not result[0]:
            return None
        
        # get all the found text
        for line in result[0]:
            text = line[1][0].strip()
            confidence = line[1][1]
        # keep only text that could be jersey numbers
        if confidence > 0.7 and text.isdigit() and 1 <= int(text)<=99:
            return int(text)
        
        return None
    

    def update(self, frame, detections):
        # for each detection try to read the number
        # majority vote is used to confirm it

        for det in detections:
            track_id = det["id"]
            if track_id < 0:
                continue
        

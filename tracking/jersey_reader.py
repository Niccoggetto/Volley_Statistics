import os

from matplotlib.pyplot import gray
from paddle import crop
os.environ["FLAGS_use_mkldnn"] = "0" # disable MKL-DNN to avoid potential issues with PaddleOCR on some systems
os.environ["FLAGS_enable_pir_in_executor"] = "0"  # disable pir in executor to avoid potential issues with PaddleOCR on some systems

import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict, Counter
import torch


class JerseyReader:
    def __init__(self):
        # Initialize the OCR model only for numbers, language eng
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_gpu=torch.cuda.is_available(),
            lang="en",
            show_log=False,
        )
        print(f"Using OCR device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        self.readings = defaultdict(list) # store OCR readings for each player ID to improve accuracy with multiple readings
        self.min_readings = 5 # minimum number of readings required to determine a jersey number
        self.confirmed =  {} # store confirmed jersey numbers for each player ID


    def extract_jersey_crop(self, frame, bbox, crop_ratio = 0.5):
        # extract a crop of the jersey area from the bounding box, we will focus on the upper part of the bounding box where the jersey number is usually located
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        # ignore bounding boxes too little
        if height < 60 or width < 30:
            return None

        # to get only the upper portion of the bbox:
        crop_y2 = y1 + int(height * crop_ratio)
        crop = frame[y1:crop_y2, x1:x2]

        if crop.size == 0:
            return None
        return crop
    

    def preprocess_crop(self, crop):
        # preprocess the crop for better OCR results
        crop = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC) # upscale the image for better OCR accuracy
        # conversion to grayscale and thresholding to enhance the numbers
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # apply higher contrast thresholding to make numbers stand out more
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        # reduce noise with a median blur
        gray = cv2.medianBlur(gray, 3)
        # apply adaptive thresholding to further enhance the numbers
        binary_inv = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return binary, binary_inv
    
    '''
    def read_number(self, crop):
        # execute the OCR reading on each crop
        # PaddleOCR v3 requires a 3-channel BGR image
        binary, binary_inv = self.crops

        for crop in [binary, binary_inv]: # try both normal and inverted binary images to improve chances of reading the number correctly

            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            result = self.ocr.ocr(crop, cls=False)

            if not result or not result[0]:
                continue # no text detected, try the next crop
            
            # collect all detected text in the crop
            for line in result[0]:
                text = line [1][0].strip()
                confidence = line [1][1]

                # filter out non-digit results and low confidence readings
                if text.isdigit() and confidence > 0.7 and 1 <= int(text) <= 99: # jersey numbers are usually between numbers 1 and 99
                    return int(text)
            
        return None
    '''

    def read_number(self, crop):
        # try both normal and inverted binary images to improve chances of reading the number correctly
        binary_inv = cv2.bitwise_not(crop)

        for img in [crop, binary_inv]:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            result = self.ocr.ocr(img, cls=False)

            if not result or not result[0]:
                continue

            for line in result[0]:
                text = line[1][0].strip()
                confidence = line[1][1]

                if text.isdigit() and confidence > 0.7 and 1 <= int(text) <= 99:
                    return int(text)
            
        return None



    def update(self, frame, detections):
        # for each detection try to read the number and use majeroity voting to confirm the jersey number for each player ID
        for det in detections:
            track_id = det["track_id"]
            if track_id < 0:
                continue # skip detections without a valid track ID

            if track_id in self.confirmed:
                continue # already confirmed jersey number for this track ID

            # crop the jersey area from the frame
            crop = self.extract_jersey_crop(frame, det['bbox'])
            if crop is None:
                continue

            # preprocess the crop for better OCR results and read
            processed = self.preprocess_crop(crop)
            number = self.read_number(processed)

            if number is not None:
                self.readings[track_id].append(number)

                # if we have enough readings, determine the most common number for this track ID
                if len(self.readings[track_id]) >= self.min_readings:
                    most_common = Counter(self.readings[track_id]).most_common(1)[0]
                    jersey_number, count = most_common

                    # the number is confirmed if it appears in the majority of readings, more than 50% of times
                    if count >= self.min_readings // 2:
                        self.confirmed[track_id] = jersey_number
                        print(f"✅ Player ID:{track_id} → Jersey #{jersey_number}")
    

    def get_jersey(self, track_id):
        # get the confirmed jersey number for a given track ID
        return self.confirmed.get(track_id, None)
    

    def get_all_confirmed(self):
        # get all confirmed jersey numbers with format {track_id: jersey_number}
        return dict(self.confirmed)
import os
import cv2

INPUT_DIR = '/storage/prep/'
OUTPUT_DIR = '/storage/prep_resized/'
dir = os.listdir(INPUT_DIR)

for fn in dir:
    proc_img(INPUT_DIR, OUTPUT_DIR, fn)
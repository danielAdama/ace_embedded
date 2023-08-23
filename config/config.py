import os
from ultralytics import YOLO
import numpy as np
import cv2
import torch

VIDEO_PATH = os.path.join(os.getcwd(),'video')
VIDEO = os.path.join(VIDEO_PATH, 'footage.mp4')
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(10)]
ALLOWED_IDS = [0, 2]
CONFI_THRESH = 0.5
MODEL = YOLO("yolov8s.pt")
EMBEDDER_MODEL="mars-small128.pb"
EMBEDDER="mobilenet"
is_GPU=True
MAX_AGE=50
MAX_COS_DIST=0.3
NN_BUDGET=5
FONT=cv2.FONT_HERSHEY_SIMPLEX
LOG_FILE='output_log.txt'
IMAGE_CAPTION = "nlpconnect/vit-gpt2-image-captioning"
MAX_LEN = 14
NUM_BEAMS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
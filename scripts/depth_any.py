from icecream import ic
from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import requests
import cv2

file = '/Users/antonia/Desktop/Screenshot 2024-02-15 at 09.08.43.png'
image = Image.open(file)
ic(image)

# exit()
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
result = pipe(image)
ic(type(result["depth"]))
#show depth map using pillow
depth_map = result["depth"]
depth_map.show()

# Carry out keypoint detection on input image(s)

import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.hand import Hand

from google.colab.patches import cv2_imshow

hand_estimation = Hand('model/hand_pose_model.pth')
oriImg = cv2.imread(test_image)  # B,G,R order

if crop=='manual':
  # bounding box for hand - hardcoded
  x = 1250
  y = 690
  w = 512
  cropped_hand = oriImg[y:y+w, x:x+w, :] # input for keypoint detection
elif crop=='auto': pass
else: cropped_hand = oriImg

'''
if image_processing:
  # test parameters here
  output_hand = cv2.resize(cropped_hand, (512,512))
  output_hand = cv2.blur(output_hand, (8,8))
  #output_hand = cv2.flip(output_hand, 1)

else: output_hand = cropped_hand

# detect hand keypoints
peaks = hand_estimation(output_hand)
all_hand_peaks.append(peaks)

# display image of hand with indicated detected keypoints
canvas = copy.deepcopy(output_hand)
canvas = util.draw_handpose(canvas, all_hand_peaks)

if crop != 'False' and show_crop:
  # display original image with cropping/bounding box - debugging only
  cv2.rectangle(oriImg, (x, y), (x+w, y+w), (0, 255, 0, 255), 2)
  cv2_imshow(oriImg) #cv.imshow 'causes Jupyter sessions to crash' in Colab

print(peaks)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
'''
sizes = [(128,128), (256,256), (512,512), (768,768), (1024,1024)]
blurs = [(1,1),(3,3),(5,5),(7,7),(9,9)]

for s in sizes:
  for b in blurs:
    all_hand_peaks = [] # format appropriate for canvas drawing
    
    # test parameters here
    resized_hand = cv2.resize(cropped_hand, s)
    output_hand = cv2.blur(resized_hand, b)

    # detect hand keypoints
    peaks = hand_estimation(output_hand)
    all_hand_peaks.append(peaks)

    # display image of hand with indicated detected keypoints
    canvas = copy.deepcopy(output_hand)
    canvas = util.draw_handpose(canvas, all_hand_peaks)

    # display original image with cropping/bounding box - debugging only
    #cv2.rectangle(oriImg, (x, y), (x+w, y+w), (0, 255, 0, 255), 2)
    #cv2_imshow(oriImg) #cv.imshow 'causes Jupyter sessions to crash' in Colab

    print('size:', s, 'blur:', b)
    print(peaks)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()

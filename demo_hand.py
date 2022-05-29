import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.hand import Hand

from google.colab.patches import cv2_imshow

hand_estimation = Hand('model/hand_pose_model.pth')
#test_image = 'input/hand_real.png'
test_image = 'output_images/out301.png'
oriImg = cv2.imread(test_image)  # B,G,R order

# bounding box for hand - hardcoded
x = 525
y = 1050
w = 512

all_hand_peaks = [] # format appropriate for canvas drawing
# loop removed - assumes only 1 hand exists
cropped_hand = oriImg[y:y+w, x:x+w, :] # input for keypoint detection

# test parameters here
output_hand = cv2.resize(cropped_hand, (128,128))
output_hand = cv2.blur(cropped_hand, (15,15))

# detect hand keypoints
peaks = hand_estimation(output_hand)
all_hand_peaks.append(peaks)

# display image of hand with indicated detected keypoints
canvas = copy.deepcopy(output_hand)
canvas = util.draw_handpose(canvas, all_hand_peaks)

# display original image with cropping/bounding box - debugging only
#cv2.rectangle(oriImg, (x, y), (x+w, y+w), (0, 255, 0, 255), 2)
#cv2_imshow(oriImg) #cv.imshow 'causes Jupyter sessions to crash' in Colab

print(peaks)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()

'''
sizes = [(128,128), (256,256), (512,512), (768,768), (1024,1024)]
blurs = [(13,13),(17,17),(21,21)]#[(1,1), (3,3), (5,5), (7,7), (9,9)]

for s in sizes:
  for b in blurs:
# test parameters here
    output_hand = cv2.resize(cropped_hand, s)
    output_hand = cv2.blur(cropped_hand, b)

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
'''

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

data_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for book_dir in sorted(os.listdir(data_dir)):
    if book_dir != '.DS_Store':
        print(book_dir)
        for img_name in os.listdir(data_dir + book_dir + '/'):
            print(img_name)
            if img_name != '.DS_Store':
                img = cv2.imread(data_dir + book_dir + '/' + img_name, 0)
                img_shape = img.shape
                img = img[int(0.05 * img_shape[0]):(img_shape[0] - int(0.1 * img_shape[0])), int(0.05 * img_shape[1]):(img_shape[0] - int(0.2 * img_shape[1]))]
                bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 23)

                blurred = cv2.medianBlur(bin_img, 5)
                img_dilation = cv2.dilate(blurred, np.ones((7, 7), np.uint8), iterations=1)

                bin_img = np.bitwise_and(bin_img, img_dilation)
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
                bin_img = cv2.medianBlur(bin_img, 3)

                out_path = output_dir + book_dir + '/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                
                cv2.imwrite(out_path + img_name, np.bitwise_not(bin_img))
                # plt.imshow(bin_img)
                # plt.show()
            




import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

1232
1027
CORNER1 = (1082, 887)
CORNER2 = (1382, 1157)


def crop(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    file_name = file_name[:-4] + "_center.png"
    cv2.imwrite(file_name, img[CORNER1[1]:CORNER2[1], CORNER1[0]:CORNER2[0], :],
                (cv2.IMWRITE_PNG_COMPRESSION, 9))

    return file_name


def frame(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    color = (0, 0, 255, 255)
    cv2.rectangle(img, CORNER1, CORNER2, color, thickness=3)
    file_name = file_name[:-4] + "_frame.png"
    cv2.imwrite(file_name, img, (cv2.IMWRITE_PNG_COMPRESSION, 9))

    return file_name


def create_diff_image(img_name, ref_img_name):
    ref_img = cv2.imread(ref_img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    diff_img = np.zeros((img.shape[0],img.shape[1],1),np.int16)
    diff_img[:,:,0] = np.linalg.norm(ref_img, axis=2) - np.linalg.norm(img, axis=2)
    file_name = img_name[:-4] + "_diff.png"
    cv2.imwrite(file_name, (diff_img + np.abs(np.min(diff_img))).astype(np.uint8), (cv2.IMWRITE_PNG_COMPRESSION, 9))

    temp = diff_img
    plt.figure()
    plt.imshow(temp[:,:,0], interpolation='nearest', vmin=-70, vmax=70, cmap='seismic')
    plt.colorbar()
    file_name = file_name[:-4] + "_heatmap.png"
    plt.savefig(file_name)
    plt.close()

    plt.figure()
    plt.hist(diff_img.flatten(), bins=10, range=(-70, 70))
    file_name = file_name[:-len("_heatmap.png")] + "_histogram.png"
    plt.savefig(file_name)
    plt.close()

    return file_name

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cropped = crop(sys.argv[1])
    else:
        file_list = ["png.png",
                     "jp2_1000.png",
                     "jp2_100.png",
                     "jp2_10.png",
                     "jp2_1.png",
                     "jp2_5.png",
                     "jp2_4.png"]

        for file_name in file_list:    
            cropped = crop(file_name)
            framed = frame(file_name)
            diffed = create_diff_image(file_name, "png.png")
            diffed = create_diff_image(cropped, "png_center.png")

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


CORNER1 = (1100, 800)
CORNER2 = (1500, 1200)


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
    plt.imshow(temp[:,:,0], interpolation='nearest', cmap='seismic', vmin=-80, vmax=80)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel("L2-norm difference", rotation=270)
    plt.title("Difference image")
    plt.axis("off")
    file_name = file_name[:-4] + "_heatmap.png"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

    plt.figure()
    histo_vec = diff_img.flatten()
    histo_vec = histo_vec[histo_vec != 0]
    plt.hist(histo_vec, bins=161, range=(-80, 80))
    plt.title("Difference histogram")
    plt.xlabel("Number of pixels")
    plt.ylabel("L2-norm difference")
    plt.legend([f"Changed pixels: {len(histo_vec)}\nPercentage: {len(histo_vec)/len(temp.flatten())*100:.1f} %"])
    file_name = file_name[:-len("_heatmap.png")] + "_histogram.png"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

    return file_name

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cropped = crop(sys.argv[1])
        framed = frame(sys.argv[1])
        diffed = create_diff_image(sys.argv[1], "png.png")
        diffed = create_diff_image(cropped, "png_center.png")
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

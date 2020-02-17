import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


CORNER1 = (1100, 800)
CORNER2 = (1500, 1200)


def crop(file_name):
    print("CROP: ", file_name)
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    file_name = file_name[:-4] + "_center.png"
    cv2.imwrite(file_name, img[CORNER1[1]:CORNER2[1], CORNER1[0]:CORNER2[0], :],
                (cv2.IMWRITE_PNG_COMPRESSION, 9))

    return file_name


def frame(file_name):
    print("FRAME: ", file_name)
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    color = (0, 0, 255, 255)
    cv2.rectangle(img, CORNER1, CORNER2, color, thickness=3)
    file_name = file_name[:-4] + "_frame.png"
    cv2.imwrite(file_name, img, (cv2.IMWRITE_PNG_COMPRESSION, 9))

    return file_name


def create_diff_image(img_name, ref_img_name):
    print("DIFF: ", img_name, ref_img_name)
    ref_img = cv2.imread(ref_img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    temp = np.zeros((img.shape[0],img.shape[1],1))
    temp[:,:,0] = ref_img_gray.astype(np.float32) - img_gray.astype(np.float32)
    diff_img = np.linalg.norm(temp, axis=2).astype(np.uint8)
    #diff_2 = np.abs(temp[:,:,0]).astype(np.uint8)
    #test = (diff_img == diff_2)
    #print(test.any(), test.all(), test.shape)

    file_name = img_name[:-4] + "_diff.png"
    cv2.imwrite(file_name, diff_img, (cv2.IMWRITE_PNG_COMPRESSION, 9))

    temp = diff_img
    plt.figure()
    plt.imshow(temp, interpolation='nearest', cmap='gray', vmin=0, vmax=50)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel("L2-Norm Difference", rotation=270)
    plt.title(f"Difference Image - {' '.join(img_name[:-4].upper().split('_')[0:2])}")
    plt.axis("off")
    file_name = file_name[:-4] + "_heatmap.png"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(temp, interpolation='nearest', cmap='YlOrRd', vmin=0,vmax=np.max(np.abs(temp)))
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel("L2-Norm Difference", rotation=270)
    plt.title(f"Difference Image - {' '.join(img_name[:-4].upper().split('_')[0:2])}")
    plt.axis("off")
    file_name = file_name[:-len("_heatmap.png")] + "_heatmap_rel.png"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

    plt.figure()
    histo_vec = diff_img.flatten()
    histo_vec = histo_vec[histo_vec != 0]
    plt.hist(histo_vec, bins=51, range=(0, 50))
    plt.title(f"Difference Histogram - {' '.join(img_name[:-4].upper().split('_')[0:2])}")
    plt.xlabel("L2-Norm Difference")
    plt.ylabel("Number of Pixels")
    plt.legend([f"Changed Pixels: {len(histo_vec)}\nPercentage: {len(histo_vec)/len(temp.flatten())*100:.1f} %"])
    file_name = file_name[:-len("_heatmap_rel.png")] + "_histogram.png"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

    return file_name

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1][-3:] == "png":
            cropped = crop(sys.argv[1])
            framed = frame(sys.argv[1])
            diffed = create_diff_image(sys.argv[1], "png.png")
            diffed = create_diff_image(cropped, "png_center.png")
        elif sys.argv[1].find("set") != -1:
            folder = sys.argv[1] + "/"
            file_list = ["png.png",
                     "jp2_1000.png",
                     "jp2_100.png",
                     "jp2_10.png",
                     "jp2_1.png",
                     "jp2_5.png",
                     "jp2_4.png"]

            for file_name in file_list:
                file_dir = folder + file_name
                cropped = crop(file_dir)
                framed = frame(file_dir)
                diffed = create_diff_image(file_dir, (folder + "png.png"))
                diffed = create_diff_image(cropped, (folder + "png_center.png"))
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

import imageio
import dxchange
import os
import cv2
import numpy as np
def compose_gif():
    img_paths = r'D:\zz\test/'
    files = os.listdir(img_paths)
    gif_images = []
    for file in files:
        img = dxchange.read_tiff(img_paths+file)
        gif_images.append(img)
    imageio.mimsave(img_paths+"test.gif",gif_images,fps=1)
if __name__ == "__main__":
    path_tomo = r'D:\zz\sheep_tomo_512shift/'
    path_lab = r'D:\zz\ss/'
    path_out = r'D:\zz\ss_out/'
    files_tomo = os.listdir(path_tomo)
    files_lab  = os.listdir(path_lab)
    for (file_tomo,file_lab) in zip(files_tomo,files_lab):
        img_tomo = dxchange.read_tiff(path_tomo + file_tomo)
        img_lab = dxchange.read_tiff(path_lab + file_lab).reshape(512, 512)
        _, img_lab = cv2.threshold(img_lab, 0.9, 1, cv2.THRESH_BINARY)
        img_out = np.zeros((512, 512, 3))
        img_tomo = np.array(img_tomo)
        img_lab = np.array(img_lab)
        img_tomo0 = img_tomo + img_lab * 142.
        img_tomo1 = img_tomo + img_lab * 22.
        img_tomo2 = img_tomo + img_lab * (-48.)
        img_out[:, :, 0] = img_tomo0
        img_out[:, :, 1] = img_tomo1
        img_out[:, :, 2] = img_tomo2
        img_out = img_out.astype(np.float32)
        dxchange.write_tiff(img_out, path_out+file_tomo)

    # compose_gif()

import numpy as np
import cv2
def load_image(filepath, size=(0,0), scale_factor=1.0):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if(size!=(0,0)):
        img_resized= cv2.resize(img,dsize=(size[1],size[0]), interpolation = cv2.INTER_CUBIC)
    else:
        img_resized=img
    img_transposed = np.transpose(img_resized, (2, 0, 1))
    img_scaled = np.divide(img_transposed, scale_factor)
    return img_scaled
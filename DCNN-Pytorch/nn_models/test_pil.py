
import PIL, PIL.Image as image, numpy as np
import torchvision, torchvision.transforms as transforms
import cv2
import data_loading.data_loaders as DL
impil = image.open('dog.jpg')
tonumpy = transforms.Lambda(lambda pil: DL.PIL2array(pil))
imnp =  tonumpy(impil)
imnp_file = cv2.imread('dog.jpg')

cv2.namedWindow("frompil",cv2.WINDOW_AUTOSIZE)
cv2.imshow("frompil",imnp)
cv2.namedWindow("fromfile",cv2.WINDOW_AUTOSIZE)
cv2.imshow("fromfile",imnp_file)
cv2.waitKey(0)
print(imnp.shape)
print(imnp[11,11])
print(imnp_file[11,11])
rs = transforms.Resize((84,150))


impilsmall = rs(impil)
imnpsmall = tonumpy(impilsmall)
print(imnpsmall.shape)
print(imnpsmall.dtype)
cv2.namedWindow("small",cv2.WINDOW_AUTOSIZE)
cv2.imshow("small",imnpsmall)
cv2.waitKey(0)


import cv2, torch, torchvision, torchvision.transforms as transforms
def toPIL(img):
    topil = transforms.ToPILImage()
    img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return topil(img_)
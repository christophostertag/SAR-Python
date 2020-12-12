import albumentations as A
from albumentations.augmentations import transforms 
import cv2  # opencv python
import matplotlib.pyplot as plt


def augment():  # https://albumentations.ai/docs/
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/
    return A.Compose([
        A.Blur(p=1),
        A.ColorJitter(brightness=0.4, contrast=0.5, p=1),
        A.Flip(p=1)
    ], p=1)


if __name__ == '__main__':
    img = cv2.imread('bus.jpg')  # bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # rgb
    transform = augment()
    
    img = transform(image=img)['image']
    plt.imshow(img)
    plt.show()
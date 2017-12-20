import math
import numpy as np
import random
import PIL
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None

import cv2
from data_processor import *
import matplotlib.pyplot as plt


def normalize(x):
    return x/255.

def randomShift(img, u=0.25, limit=4):
    if random.random() < u:
        dx = round(random.uniform(-limit,limit))  #pixel
        dy = round(random.uniform(-limit,limit))  #pixel

        height,width,channel = img.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (width, height))
    return img

def randomRotate(img, u=0.25, limit=45):
    if random.random() < u:
        angle = random.uniform(-limit,limit)  #degree
        height, width, channel = img.shape
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img = cv2.warpAffine(img, M, (width, height))

    return img


def randomBrightness(img, limit=0.5, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(0, 1)
        img = alpha*img
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img


if __name__ == '__main__':

    train_dataset = ImageDataset(datasets="data/train.p",
                                 transform=[
                                     # lambda x: randomBrightness(x,u=1),
                                 ],
                                 )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )


    fig, sub = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(hspace=.02, wspace=.001)
    sub = sub.ravel()

    image, label, indice = train_dataset[3]

    sub[0].imshow(image)
    sub[0].set_title('Before Shift')
    sub[1].imshow(randomShift(image, u=1))
    sub[1].set_title('After Shift')
    plt.savefig('Shift.png')
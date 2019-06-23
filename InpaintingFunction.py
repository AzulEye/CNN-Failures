import numpy as np
from scipy.signal import convolve2d, convolve
from PIL import Image
import PIL
from matplotlib import pyplot as plt

def inpaint(im, mask, wSize):
    imF = np.zeros_like(im)
    for chan in range(3):
        K = [0.5, 0.5]
        K = np.expand_dims(K, 0)
        K0 = K
        IM = im[:, :, chan]
        for i in range(wSize):
            K = np.array(convolve2d(K, K0))
        imB = convolve2d(IM * mask, K, 'same')
        imB = convolve2d(imB, K.T, 'same')

        maskB = convolve2d(mask, K, 'same')
        maskB = convolve2d(maskB, K.T, 'same')

        imInt = imB / (maskB + (maskB == 0))
        imF[:, :, chan] = mask * IM + (1 - mask) * imInt
    return imF


def inpainting(im, randCropX, randCropY, h,w):
    mask = np.zeros_like(im)
    mask[randCropY:randCropY + h, randCropX:randCropX + w,
    0:3] = 1
    mask = mask[:, :, 0]
    image_resultBW = im[:, :, 0] + im[:, :, 1] + im[:, :, 2]
    o = 0
    while (np.sum(np.sum(image_resultBW == 0)) > 200 and o < 80):
        o += 1
        im = inpaint(im, mask, 10)
        image_resultBW = im[:, :, 0] + im[:, :, 1] + im[:, :, 2]
        mask = np.ones_like(image_resultBW)

        mask[image_resultBW == 0] = 0

    return im



FinalIM = np.zeros((224, 224, 3))
FinalIMBeforeInpainting = np.zeros((224, 224, 3))
sizeh = 120
sizew = 100
embedh = 50
embedw = 70
embeddedIM = Image.open('ILSVRC2012_val_00010739.JPEG')
embeddedIM = embeddedIM.resize((sizeh, sizew), PIL.Image.ANTIALIAS)
FinalIMBeforeInpainting[embedw:embedw + sizew, embedh:embedh + sizeh, :] = embeddedIM
FinalIM = inpainting(FinalIMBeforeInpainting, embedh, embedw, sizew,sizeh)

plt.subplot(121)
plt.imshow(FinalIMBeforeInpainting.astype('uint8'))
plt.subplot(122)
plt.imshow(FinalIM.astype('uint8'))
plt.show()
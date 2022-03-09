import torch
from PIL import Image
import numpy as np


def main():
    img1 = Image.open('../data/imgs/caijunyan00005.jpg')
    img1 = img1.resize(img1.size, resample=Image.NEAREST)
    img1 = np.asarray(img1)
    pass


if __name__ == '__main__':
    main()

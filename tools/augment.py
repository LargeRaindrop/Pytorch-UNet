import os

import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from tqdm import tqdm

imgs_dir = '../data/train/imgs'
masks_dir = '../data/train/masks'
imgs_aug_dir = '../data/train/aug-imgs'
masks_aug_dir = '../data/train/aug-masks'
colormap = [0,0,0] + [255,0,0] + [0,255,0] + [0,0,255] + [255,255,255] * 252
aug_methods = ['Fliplr', 'Flipud', 'CropAndPad', 'Affine', 'PiecewiseAffine', 'GaussianBlur']


def main():
    names = os.listdir(imgs_dir)
    names = [name.split('.')[0] for name in names]

    for name in tqdm(names):
        img_dir = os.path.join(imgs_dir, name + '.jpg')
        mask_dir = os.path.join(masks_dir, name + '.png')
        img = np.array(Image.open(img_dir))[:, :, :1]
        img = np.expand_dims(img, axis=0).astype(np.float32)
        mask = np.array(Image.open(mask_dir))
        mask = np.expand_dims(mask, axis=(0, 3)).astype(np.int8)

        for aug_method in aug_methods:
            func_name = 'iaa.{}()'.format(aug_method)
            img_aug, mask_aug = eval(func_name)(images=img, segmentation_maps=mask)

            img_aug = np.repeat(img_aug, 3, axis=3)
            img_aug = np.squeeze(img_aug)
            img_aug = Image.fromarray(img_aug.astype(np.uint8))
            mask_aug = np.squeeze(mask_aug)
            mask_aug = Image.fromarray(mask_aug.astype(np.uint8))
            mask_aug.putpalette(colormap)

            img_aug_dir = os.path.join(imgs_aug_dir, '{}-{}.jpg'.format(name, aug_method))
            mask_aug_dir = os.path.join(masks_aug_dir, '{}-{}.png'.format(name, aug_method))
            img_aug.save(img_aug_dir)
            mask_aug.save(mask_aug_dir)


if __name__ == '__main__':
    main()

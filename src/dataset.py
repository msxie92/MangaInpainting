import os
import glob
import scipy
import torch
import random
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread, imsave, imresize
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
from scipy import ndimage
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, line_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.line_data = self.load_flist(line_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.line = config.LINE
        self.mask = config.MASK

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        fname = self.data[index]
        img = imread(self.data[index])/255.0
        img_gray = rgb2gray(img)

        # load line
        line = self.load_line(img_gray, index, None)/255.0

        if size != 0:
            pos = self.get_pos(img_gray, size)
            img_gray = self.crop(img_gray, pos, size)
            line = self.crop(line, pos, size)
            mask = self.load_mask(size, index, pos)
        else:
            mask = self.load_mask(size, index)

        # img = torch.from_numpy(img.transpose(2,0,1).copy()).float()
        line = torch.from_numpy(line.copy()).unsqueeze(0).float()
        img_gray = torch.from_numpy(img_gray.copy()).unsqueeze(0).float()
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()

        ih,iw = img_gray.shape[-2:]
        if size == 0:
            line = self.npad(line)
            img_gray = self.npad(img_gray)
            mask = self.npad(mask)

        # img = img * 2 - 1.0
        img_gray = img_gray * 2 - 1.0
        line = line * 2 - 1.0

        if line.mean()==-1:
            line = -1*line

        return img_gray, line, mask, [ih, iw]


    def npad(self, im, pad=128, mode='constant', value=1):
        h,w = im.shape[-2:]
        hp = (h+pad-1) //pad*pad
        wp = (w+pad-1) //pad*pad
        if mode == 'constant':
            return nnF.pad(im.unsqueeze(0), (0, wp-w, 0, hp-h), mode='constant', value=value).squeeze(0)
        else:
            return nnF.pad(im.unsqueeze(0), (0, wp-w, 0, hp-h), mode='replicate').squeeze(0)

    def get_pos(self, img, crop_size):
        h, w = img.shape[:2]
        crop = 0

        x = random.randint(crop, np.maximum(0, w - crop_size-crop))
        y = random.randint(crop, np.maximum(0, h - crop_size-crop))
        return (x, y)


    def crop(self, img, pos, size):
        oh, ow = img.shape[:2]
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img[y1:(y1 + th), x1:(x1 + tw),...]
        return img

    def load_line(self, img, index, mask):

        line = imread(self.line_data[index], mode='L')
        scale = img.shape[0]/line.shape[0]
        line = imresize(line, img.shape[:2], interp='bilinear')
        return line

    def load_mask(self, size, index, pos=None):
        imgh, imgw = size, size
        mask_type = self.mask

        mask = imread(self.mask_data[index])
        mask = rgb2gray(mask)
        mask = (mask > 0.5).astype(np.uint8)
        if pos is not None:
            mask = self.crop(mask, pos, size)
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]
        # print(img.shape)
        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist
            
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(open(flist,'r'), delimiter='', invalid_raise=False, dtype=np.str, encoding='utf-8')
                    # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

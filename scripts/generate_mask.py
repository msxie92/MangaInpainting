import argparse
import numpy as np
import random
from PIL import Image
import cv2

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='mask')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.N):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        ini_x = random.randint(0, args.image_size - 1)
        ini_y = random.randint(0, args.image_size - 1)
        mask = random_walk(canvas, ini_x, ini_y, args.image_size ** 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ret,mask = cv2.threshold(np.uint8(mask * 255),128,1,cv2.THRESH_BINARY)
        iters = np.random.randint(0,args.iter)
        mask = cv2.erode(mask, kernel, iterations=iters)
        print("save:", i, np.sum(mask)/255)

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.png'.format(args.save_dir, i))

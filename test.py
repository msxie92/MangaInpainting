import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.manga_inpaintor import MangaInpaintor

def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    config.GPU = [i for i in range(len(config.GPU))]


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = MangaInpaintor(config)
    model.load()


    print('\nstart testing...\n')
    with torch.no_grad():
        model.test()



def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: semantic inpaint model, 2: manga inpaint model, 3: manga inpaint model with fixed semantic inpaint model, 4: joint model')
    # for test mode
    parser.add_argument('--input', type=str, help='path to the manga images directory or an manga image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--line', type=str, help='path to the lines directory or a line file')
    parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    config.MODEL = args.model if args.model is not None else 4
    config.INPUT_SIZE = 0

    if args.input is not None:
        config.TEST_FLIST = args.input

    if args.mask is not None:
        config.TEST_MASK_FLIST = args.mask

    if args.line is not None:
        config.TEST_LINE_FLIST = args.line

    if args.output is not None:
        config.RESULTS = args.output

    return config


if __name__ == "__main__":
    main()

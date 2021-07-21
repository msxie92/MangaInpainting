import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import SemanticInpaintGenerator, MangaInpaintGenerator
from .svae import ScreenVAE
import itertools
import torch.nn.functional as F
import random
from .morphology import Dilation2d, Erosion2d


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')

    def load(self):
        # if os.path.exists(self.gen_weights_path):
        print('Loading %s generator...' % self.name)

        if torch.cuda.is_available():
            data = torch.load(self.gen_weights_path)
        else:
            data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

        if isinstance(self.generator, torch.nn.DataParallel):
            self.generator.load_state_dict(data['generator'])
        else:
            net=nn.DataParallel(self.generator)
            net.load_state_dict(data['generator'])
            self.generator=net.module
            del net
            
        del data


class SemanticInpaintingModel(BaseModel):
    def __init__(self, config):
        super(SemanticInpaintingModel, self).__init__('SemanticInpaintingModel', config)

        generator = SemanticInpaintGenerator(in_channels=7)

        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)

        self.add_module('generator', generator)

        self.erode = Erosion2d(1,1,3, soft_max=False)
        self.iters = 3
        self.grained = 1/config.SHRIK_ITERS

    def forward(self, screen_masked, edges_masked, masks):
        noise = torch.randn_like(masks)
        inputs = torch.cat((screen_masked, edges_masked, masks, noise), dim=1)
        output_image, output_edge = self.generator(inputs.detach())
        return output_image, output_edge

    def test(self, screen_masked, edges_masked, masks):
        output_images, output_edges = [],[]
        noise = torch.randn_like(masks)
        # grained = 0.2

        all_masks = []
        maskstl = torch.chunk(masks, masks.shape[0], dim=0)
        nmasksl = []
        for i in range(len(maskstl)):
            maskst = maskstl[i]
            masksnt = [maskst]
            for t in range(1, int(1/self.grained)):
                while maskst.sum()>maskstl[i].sum()*(1-self.grained*t):
                    maskst = self.erode(maskst, iterations=3)
                masksnt.append(maskst)
            masksnt = torch.cat(masksnt, dim=0)
            nmasksl.append(((maskstl[i]+masksnt)/2).unsqueeze(1))
        maskst = torch.cat(nmasksl, dim=1)

        for nmasks in maskst:
            inputs = torch.cat((screen_masked, edges_masked, nmasks, noise), dim=1)
            screen_maskedt, edges_maskedt = self.generator(inputs)
            output_images.append(screen_maskedt)
            output_edges.append(edges_maskedt)
            screen_masked = screen_masked*(1-masks) + screen_maskedt*masks
            edges_masked = edges_masked*(1-masks) + edges_maskedt*masks
        return output_images, output_edges, maskst


class MangaInpaintingModel(BaseModel):
    def __init__(self, config):
        super(MangaInpaintingModel, self).__init__('MangaInpaintingModel', config)

        in_channels = 6
        self.cnum = 32 
        generator = MangaInpaintGenerator(in_channels, self.cnum)

        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)

        self.add_module('generator', generator)

    def forward(self, images, hints, masks):
        images_masked = (images * (1 - masks).float()) #+ masks
        inputs = torch.cat((images_masked, hints), dim=1)
        outputs = self.generator(inputs, masks)                                    # in: [rgb(3) + edge(1)]
        return outputs

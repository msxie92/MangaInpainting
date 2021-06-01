import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import SemanticInpaintingModel, MangaInpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .svae import ScreenVAE
import torch.nn.functional as F
from .morphology import Dilation2d, Erosion2d


class MangaInpaintor():
    def __init__(self, config):
        self.config = config

        self.semantic_inpaint_model = SemanticInpaintingModel(config).to(config.DEVICE)
        self.manga_inpaint_model = MangaInpaintingModel(config).to(config.DEVICE)
        self.svae_model = ScreenVAE().to(config.DEVICE)

        self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_LINE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

    def load(self):
        print('Loading models...')
        self.semantic_inpaint_model.load()
        self.manga_inpaint_model.load()

    def test(self):
        self.semantic_inpaint_model.eval()
        self.manga_inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            print(index, name)
            images, lines, masks = self.cuda(*items[:3])
            index += 1

            # self.save_images(lines[:,:,:h,:w], name, 'lines')
            # self.save_images(images[:,:,:h,:w], name, 'manga')

            h,w = items[3]
            # if lines.shape[2]*lines.shape[3]>1280*1024:
            #     hp,wp = 1280,1024
            #     masks = masks[:,:,:hp,:wp]
            #     lines = lines[:,:,:hp,:wp]
            #     images = images[:,:,:hp,:wp]

            dilate = Dilation2d(1,1,3, soft_max=False)
            # self.save_images(masks[:,:,:h,:w]*2-1, name, 'masks')
            masks = dilate(masks, iterations=2)
            manga_masked = (images * (1 - masks)) + masks
            lines_masked = (lines * (1 - masks)) + masks

            screen_masked = self.svae_model(manga_masked, lines_masked, rep=True)
            screen0 = self.svae_model(images, lines, rep=True)
            # self.save_images(screen0[:,:,:h,:w], name, 'screen')

            manga_masked = (images * (1 - masks)) + masks
            lines_masked = (lines * (1 - masks)) + masks

            # self.save_images(manga_masked[:,:,:h,:w], name, 'manga_masked')
            # self.save_images(lines_masked[:,:,:h,:w], name, 'lines_masked')
            # self.save_images(screen_masked[:,:,:h,:w], name, 'screen_masked')

            scale=False
            if scale:
                lines_masked = F.interpolate(lines_masked,scale_factor=0.5, mode='bilinear')
                screen_masked = F.interpolate(screen_masked, scale_factor=0.5, mode='bilinear')
                masks = F.interpolate(masks, scale_factor=0.5, mode='nearest')

            screenl, linesl, masksl = self.semantic_inpaint_model.test(screen_masked, lines_masked, masks)

            t = 0
            for screen, lines, maskst in zip(screenl, linesl, masksl):
                t = t+1
                self.save_images(lines[:,:,:h,:w], name[:-4]+'_%d.png'%t, 'lines_decode')
                self.save_images(screen[:,:,:h,:w], name[:-4]+'_%d.png'%t, 'screen_decode')
                self.save_images(maskst[:,:,:h,:w]*2-1, name[:-4]+'_%d.png'%t, 'masks')
                # lines[lines>0]=1
                screen_decode = screen_masked *(1-masks) + screen * masks
                lines_decode = lines_masked *(1-masks) + lines * masks
                # manga_recons = self.svae_model(screen_decode, lines_decode, screen=True)
                # manga_recons = images *(1-masks) + manga_recons * masks
                # self.save_images(manga_recons[:,:,:h,:w], name[:-4]+'_%d.png'%t, 'manga_recons')
                if scale:
                    screenr = F.interpolate(screen, scale_factor=2, mode='bilinear')
                    linesr = F.interpolate(lines, scale_factor=2, mode='bilinear')
                    masksr = F.interpolate(masks, scale_factor=2, mode='bilinear')
                    outputs = self.manga_inpaint_model(images, torch.cat([screenr, linesr],1), masksr)
                    outputs_merged = (outputs * masksr) + (images * (1 - masksr))
                    outputs_merged_l = (outputs_merged + 1)*(linesr+1)/2 -1
                else:
                    outputs = self.manga_inpaint_model(images, torch.cat([screen, lines],1), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))
                    outputs_merged_l = (outputs_merged + 1)*(lines+1)/2 -1

                # self.save_images(outputs[:,:,:h,:w], name[:-4]+'_%d.png'%t, 'manga_decode')
                self.save_images(outputs_merged[:,:,:h,:w], name[:-4]+'_%d.png'%t, 'manga_merged')
                # self.save_images(outputs_merged, name[:-4]+'_%d.png'%t, 'manga_merged')

            torch.cuda.empty_cache()

        print('\nEnd test....')

    def save_images(self, img, name, fld_name):
        # if img.shape[1] > 3:
        #     img = img[:,:3,:,:]
        output = self.postprocess(img)[0]
        os.makedirs(os.path.join(self.results_path, fld_name), exist_ok=True)
        path = os.path.join(self.results_path, fld_name, name)
        imsave(output, path)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        # img = img * 255.0
        img = img * 127.5+127.5
        img = img.permute(0, 2, 3, 1)
        return img.int()

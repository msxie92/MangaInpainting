## Seamless Manga Inpainting with Semantics Awareness
### [SIGGRAPH 2021](https://dl.acm.org/doi/10.1145/3450626.3459822) | [Project Website](https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html) | [BibTex](#citation)

### Introduction:
Manga inpainting fills up the disoccluded pixels due to the removal of dialogue balloons or ``sound effect'' text. This process is long needed by the industry  for the language localization  and the conversion to animated manga. It is mostly done manually, as existing methods (mostly for natural image inpainting) cannot produce satisfying results. 
We present the first manga inpainting method, a deep learning model, that generates high-quality results. Instead of direct inpainting, we propose to separate the complicated inpainting into two major phases, semantic inpainting and appearance synthesis. This separation eases both the feature  understanding and  hence the training of the learning model. A key idea is to disentangle the structural line and screentone, that helps the network to better distinguish the structural line and the screentone features for semantic interpretation. 
Detailed description of the system can be found in our [paper](https://www.cse.cuhk.edu.hk/~ttwong/papers/mangainpaint/mangainpaint.html).

<!-- ------------------------------------------------------------------------------ -->
## Example Results 
Belows shows an example of our inpainted manga image. Our method automatically fills up the disoccluded regions with meaningful structural lines and seamless screentones.
![Example](examples/representative.png)

<!-- ------------------------------------------------------------------------------ -->
## Prerequisites
- Python 3.6
- PyTorch 1.2
- NVIDIA GPU + CUDA cuDNN

<!-- ------------------------------------------------------------------------------ -->
## Installation
- Clone this repo:
```bash
git clone https://github.com/msxie92/MangaInpainting.git
cd MangaInpainting
```
- Install PyTorch and dependencies from http://pytorch.org
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets
### 1) Images
As most of our training manga images are under copyright, we recommend you to use restored [Manga109 dataset](http://www.manga109.org/en/). 
Please download datasets from official websites and then use [Manga Restoration](https://github.com/msxie92/MangaRestoration) to restored the bitonal nature. 
Please use a larger resolution instead of the predicted one to tolerant the prediction error. Exprically, set scale>1.4. 

### 2) Structural lines
Our model is trained on structural lines extracted by [Li et al.](https://www.cse.cuhk.edu.hk/~ttwong/papers/linelearn/linelearn.html). You can download their publically available [testing code](https://github.com/ljsabc/MangaLineExtraction).

### 3) Masks
Our model is trained on both regular masks (randomly generated rectangle masks) and irregular masks (provided by [Liu et al. 2017](https://arxiv.org/abs/1804.07723)). You can download publically available Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).
Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

[MangaInpainting](https://drive.google.com/file/d/1YeVwaNfchLhy3lAA7jOLBP-W23onjy8S/view?usp=sharing)

[ScreenVAE](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view)

### Testing
In each case, you need to provide an input image, a line drawing image and a mask image. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py --checkpoints [path to checkpoints] \
      --input [path to the output directory]\
      --mask [path to the output directory]\
      --line [path to the output directory]\
      --output [path to the output directory]
```

We provide some test examples under `./examples` directory. Please download the [pre-trained models](#getting-started) and run:
```bash
python test.py --checkpoints ./checkpoints/mangainpaintor \
      --input examples/test/imgs/ \
      --mask examples/test/masks/ \
      --line examples/test/lines/ \
      --output examples/test/results/
```
This script will inpaint all images in `./examples/test/imgs` using their corresponding masks in `./examples/test/mask` directory and saves the results in `./examples/test/results` directory. 

### Model Configuration
The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. 

## Copyright and License
You are granted with the [`LICENSE`](LICENSE) for both academic and commercial usages.

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:

```
@article{xie2021seamless,
        title    ={Seamless Manga Inpainting with Semantics Awareness},
        author   ={Minshan Xie and Menghan Xia and Xueting Liu and Chengze Li and Tien-Tsin Wong},
        journal  = {ACM Transactions on Graphics (SIGGRAPH 2021 issue)},
        month    = {August},
        year     = {2021},
        volume   = {40},
        number   = {4},
        pages    = {96:1--96:11}
}
```

## Reference
- [ScreenVAE](https://github.com/msxie92/ScreenStyle)
- [Edge-Connect](https://github.com/knazeri/edge-connect)
- [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch)

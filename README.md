# MMLab with ActNN

In this repo, we will show how to integrate ActNN for various computer vision tasks with MMLab. There are basically five steps to go as below, and we will do it step by step.

0. Install OpenMMLab libraries
1. Install ActNN package
2. Implement ActNN hook
3. Modify config file to use it
4. Train and test a model

The key files are listed as below

```
actnn-mim
├── README.md
├── configs
│   └── resnet50_b64x4_imagenet.py
│   └── vgg11bn_b64x4_imagenet.py
├── data
│   └── imagenet
│   │   ├── train
│   │   └── val
├── work_dirs
│   ├── resnet50_b64x4_imagenet
│   │   ├── resnet50_b64x4_imagenet.log
│   │   ├── resnet50_b64x4_imagenet.pth
│   │   └── resnet50_b64x4_imagenet.py
│   └── vgg11bn_b64x4_imagenet
│       ├── vgg11bn_b64x4_imagenet.log
│       ├── vgg11bn_b64x4_imagenet.pth
│       └── vgg11bn_b64x4_imagenet.py
└── hook_actnn.py
```

## Install OpenMMLab libraries

> **Requirements**
> - Python 3.6+
> - PyTorch 1.10+

```bash
conda create -n mim python=3.7 -y && conda activate mim
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install openmim
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
mim install mmcls -f https://github.com/open-mmlab/mmclassification.git
mim install mmdet -f https://github.com/open-mmlab/mmdetection.git
mim install mmseg -f https://github.com/open-mmlab/mmsegmentation.git
```

## Install ActNN package

```bash
git clone https://github.com/ucbrise/actnn.git
cd actnn/actnn
pip install -v -e .
```

## Implement a new dataset class

Then we need to implement a new hook class `ActNNHook`, the key implementation of the class is as below.

```python
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

import actnn
import torch

@HOOKS.register_module()
class ActNNHook(Hook):

    def __init__(self, default_bit=4):
        self.default_bit = default_bit

    def after_run(self, runner):
        torch._C._autograd._reset_saved_tensors_default_hooks()

    def before_run(self, runner):
        controller = actnn.controller.Controller(
            default_bit=self.default_bit)

        def pack_hook(x):
            r = controller.quantize(x)
            del x
            return r

        def unpack_hook(x):
            r = controller.dequantize(x)
            del x
            return r

        torch._C._autograd._register_saved_tensors_default_hooks(
            pack_hook, unpack_hook)
        runner.controller = controller

    def before_train_epoch(self, runner):
        model = (runner.model.module if is_module_wrapper(
            runner.model) else runner.model)
        runner.controller.unrelated_tensors = set()
        runner.controller.filter_tensors(model.named_parameters())

    def after_train_iter(self, runner):
        runner.controller.iterate()
```

## Modify Config

The key step is to define the `custom_imports` so that MMCV will import the file specified in the list of `imports` when loading the config.
This will load the file `hook_actnn.py` we implemented in the previous step
so that the `ActNNHook` can be registered into the `HOOKS` registry in MMLab package.

```python
custom_imports = dict(
    imports=['hook_actnn'],
    allow_failed_imports=False)
```

The whole configs to run a model can be found in `configs/*.py` which defined the model and dataset to run.

## Train and test the model

Finally, we can train and evaluate the model through the following command

```bash
PYTHONPATH=$PWD:$PYTHONPATH mim train mmcls configs/resnet50_b64x4_imagenet.py --launcher pytorch --gpus 4 --work-dir work_dirs/resnet50_b64x4_imagenet

PYTHONPATH=$PWD:$PYTHONPATH mim train mmcls configs/vgg11bn_b64x4_imagenet.py --launcher pytorch --gpus 4 --work-dir work_dirs/vgg11bn_b64x4_imagenet

PYTHONPATH=$PWD:$PYTHONPATH mim train mmdet configs/retinanet_r50_fpn_b4x4_coco.py --launcher pytorch --gpus 4 --work-dir work_dirs/retinanet_r50_fpn_b4x4_coco

PYTHONPATH=$PWD:$PYTHONPATH mim train mmseg configs/fcn_r18-d8_512x1024_b8x1_cityscapes.py --launcher pytorch --gpus 1 --work-dir work_dirs/fcn_r18-d8_512x1024_b8x1_cityscapes
```

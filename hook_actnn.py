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

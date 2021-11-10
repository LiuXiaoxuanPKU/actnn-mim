# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hook import HOOKS, Hook

import actnn

@HOOKS.register_module()
class ActNNHook(Hook):
    """.
    This hook will call actnn controller.iterate, which is used to update auto precision
    Args:
        actnn (bool): If actnn is enabled
        interval (int): Update interval (every k iterations)
    """

    def __init__(self, interval=1, default_bit=4, auto_prec=False):
        self.interval = interval
        self.default_bit = default_bit
        self.auto_prec = auto_prec

    def before_run(self, runner):
        runner.controller = actnn.controller.Controller(
            default_bit=self.default_bit, auto_prec=self.auto_prec)
    
    def before_train_epoch(self, runner):
        runner.controller.unrelated_tensors = set()
        runner.controller.filter_tensors(runner.model.named_parameters())

    def after_train_iter(self, runner):
        if self.auto_prec and self.every_n_iters(runner, self.interval):
            model = (
                runner.model.module if is_module_wrapper(
                    runner.model) else runner.model
            )
            runner.controller.iterate(model)

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

import pickle
import torch

@HOOKS.register_module()
class SaveGradientHook(Hook):

    def __init__(self, out_file):
        self.out_file = out_file

    def before_run(self, runner):
        import random
        random.seed(0)
        import numpy as np
        np.random.seed(0)
        # torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)

    def after_train_iter(self, runner):
        model = (runner.model.module if is_module_wrapper(
            runner.model) else runner.model)

        params = list(model.parameters())

        grad = None
        for param in params:
            cur_grad = torch.flatten(param.grad)
            if grad is None:
                grad = cur_grad
            else:
                grad = torch.cat([grad, cur_grad])

        with open(self.out_file, 'wb') as f:
            pickle.dump(grad, f)
        exit(0)


    
        

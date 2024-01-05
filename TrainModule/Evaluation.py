from collections import OrderedDict
from tqdm import tqdm
import torch
from .progress import ProgressBar
from glob import glob


class Eval():
    '''
    self.lr
    self._epoch
    self.step
    '''

    def __init__(self,
                 task,
                 device,
                 ckpt_dict,
                 ):


        self.task = task
        self.ckpt_dict = ckpt_dict
        self.device = device

        self.evalLoader = task.test_dataloader()
        self.device_num = device
        if device in ['cpu', 'CPU']:
            self.device = 'cpu'
        elif device in [0, 1, 2, 3]:
            self.device = 'cuda:{}'.format(device)


        for module in task.children():
            module = module.to(self.device)

        self.task.load_from_checkpoint(
                                    device = self.device,
                                    ckpt_dict = ckpt_dict
                                    )

    def run(self):
        with torch.no_grad():
            for batch in tqdm(self.evalLoader):
                batch = [x.to(self.device) for x in batch]
                self.task.evaluation_step(batch)

            self.task.evaluation_epoch_end()

        # with torch.no_grad():
        #     for batch in tqdm(range(self.valLoader.dataset.__len__())):
        #         batch = [x.to(self.device) for x in batch]
        #         self.task.evaluation_step(batch)
        #
        #     self.task.evaluation_epoch_end()

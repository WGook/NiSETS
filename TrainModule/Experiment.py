from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

class ExperimentSetup(nn.Module):
    '''
    self.model : Checkpoint로 저장되는 인스턴스



    (변경중)
    self.on_save_checkpoint: checkpoint를 저장할 모델을 return (default: self.model) -> dict
        key값은 저장되는 파일명을 결정
    '''


    def __init__(self, save_dir = None):
        self.optimizer = None
        super().__init__()
        self.model = None
        self.model_dict = None
        self.loss_func = None
        self.logs = {}
        self.save_dir = save_dir
        self.milestone = 0
        self.results_folder = None
        self.automatic_optimization = True
        self.automatic_scheduler = True
        self.optimizer = None
        self.scheduler = None
        self.amp_scaler = None
        
        if not self.automatic_scheduler:
            print('''
                  ===========***===========
                  You must define your own Class methods:
                  1. def get_lr(self):
                  2. def state_dicts_list(self, step):
                  3. def load_from_state_dicts(self, data):
                  ===========***===========
                  ''')
        # if self.model_dict is None:
        #     raise Warning("You should define self.model_dict = {'model': self.model} for save checkpoint")

    # def count_param(self):
    #     return func(self.model)

    def cycle(self, dl):
        while True:
            for data in dl:
                yield data

    def train_dataloader(self):
        return
    def val_dataloader(self):
        return
    def test_dataloader(self):
        return

    def augmentation(self, batch):
        return batch
    def configure_optimizers(self):
        '''
        self.optimizer =

        self.scheduler = {
                        'scheduler': ,
                        'interval': 'epoch' or 'step'
                        }
                        or None

        return self.optimizer, self.scheduler

        '''
        return

    def on_training_start(self):
        '''
        log all parameters what you want when start epoch
        '''
        self.log({'learning rate': self.get_lr()})
        self.lr = self.get_lr()


    def training_step(self, batch):
        # return {'loss': loss}
        return

    def before_training_step(self):
        return None

    def training_step_end(self, outputs):
        # torch.stack([x['loss'] for x in outputs]).mean()
        return None

    def validation_step(self, batch):
        # return {'loss': loss}
        return
    def validation_loop_end(self, outputs):
        # torch.stack([x['loss'] for x in outputs]).mean()
        return None

    def evaluation_step(self, batch):
        return None

    def evaluation_epoch_end(self):
        return None

    def evaluation_step_per_item(self, batch):
        '''
        batch의 각 요소의 shape에 batch size는 들어있지 않음
        즉, batch는 dataset의 get item으로 받아옴 (batch size = 1이고 squeeze된 상태)
        '''
        return None

    def evaluation_epoch_per_item_end(self):
        return None


    def state_dicts_list(self, step):
        if self.args.fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.ddp_model, StateDictType.FULL_STATE_DICT, save_policy):
                task_state = self.ddp_model.state_dict()
                amp_state = self.amp_scaler.state_dict()
                if self.scheduler is not None:
                    sch_state = self.scheduler.state_dict()
        else:
            task_state = self.state_dict()
            amp_state = self.amp_scaler.state_dict()
            if self.scheduler is not None:
                sch_state = self.scheduler.state_dict()
        
        if self.scheduler is not None:
            data = {
                'step': step,
                'task': task_state,
                'scaler': amp_state,
                'lr_scheduler': sch_state
            }
        else:
            data = {
                'step': step,
                'task': task_state,
                'scaler': amp_state
            }

        return data

    def load_from_state_dicts(self, data):
        self.load_state_dict(data['task'], strict = False)
        self.amp_scaler.load_state_dict(data['scaler'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(data['lr_scheduler'])

    def load_from_checkpoint(self, device, ckpt_dict):
        '''
        ckpt_dict : {'모델의 변수명': 모델의 checkpoint path}
        '''

        for key in ckpt_dict.keys():
            for name, child in self.named_children():
                if key == name:
                    child.load_state_dict(torch.load(ckpt_dict[key], map_location = device))



    # ===============================

    def log(self, data):
        for key in data.keys():
            self.logs[key] = data[key]


    # ===============================

    def get_lr(self):
        if self.scheduler is not None:
            return self.scheduler.get_lr()[0]
        else:
            return self.optimizer.param_groups[-1]['lr']

    # ===============================

    def debugging(self, device):
        print('\n>>> Model debugging...')
        with torch.no_grad():
            tr_loader = self.train_dataloader()
            val_loader = self.val_dataloader()
            trainList = []
            valList = []
            print('.')
            for module in self.children():
                module = module.to(device)
            for batch_idx, batch in enumerate(tr_loader):
                batch = [x.to(device) if type(x) == torch.Tensor else x for x in batch]
                trainList.append(self.training_step(batch))
                break
            self.training_epoch_end(trainList)
            print('.')

            for batch_idx, batch in enumerate(val_loader):
                batch = [x.to(device) if type(x) == torch.Tensor else x for x in batch]
                valList.append(self.validation_step(batch))
                break
            self.validation_epoch_end(valList)
            print('.')
        torch.cuda.empty_cache()
        print(' Complete.\n')

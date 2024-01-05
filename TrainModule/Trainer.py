from collections import OrderedDict
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from .progress import ProgressBar
from .checkpoint import Monitor
import os
from pathlib import Path
from time import time
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class Trainer():
    '''
    self.lr
    self._epoch
    self.step
    '''

    def __init__(self,
                 task,
                 train_num_steps,
                 device,
                 validation_step,
                 save_ckpt,
                 monitor,
                 ckpt = None,
                 logger = None,
                 inference = False):


        self.task = task
        self.train_num_steps = train_num_steps
        self.ckpt = Monitor(**monitor)
        self.save_nums = monitor['save_top_k']
        self.validation_step = validation_step
        self.save_ckpt = save_ckpt
        if logger is not None:
            self.logger = logger
            self.log_id = logger.dir.split('/')[-2].split('-')[-1]

        else:
            self.logger = None
            self.log_id = 'off_'+str(int(time()//10))
            self.ckpt.ckpt_path = os.path.join('./checkpoint', self.log_id)

        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint')
        results_folder = os.path.join('./checkpoint', self.log_id)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.task.results_folder = self.results_folder

        self.gradient_accumulate_every = 1
        self.gpu_availables = [x for x in range(torch.cuda.device_count())]

        if task.args.inference:
            for module in task.children():
                module = module.to(device)
                self.device_for_inf = device
        else:
            for module in task.children():
                module = module.to(dist.get_rank())

        if not task.args.inference:
            self.trainLoader = task.train_dataloader()
            self.valLoader = task.val_dataloader()
            self.evalLoader = task.test_dataloader()

        self.step = 0
        self.train_outputs = []
        self.val_outputs = []
        self.eval_outputs = []
        self.log_history = {}

        # task.load_from_state_dicts(torch.load('/home/gook/Local/E2ETTS/diffe2e_dis/checkpoint/anneal_dis/task-390000.pt', map_location = 'cpu'))
    def run(self):
        if dist.get_rank() == 0:
            self.summary()
        self._progress = ProgressBar(self.task, self.train_num_steps, initial = self.step)
        if self.task.automatic_optimization:
            self.task.optimizer.zero_grad()
        while self.step < self.train_num_steps:
            self.task.on_training_start()
            self.log(self.task.logs)
            self.reset_log('step')
            self._progress.on_training_start()
            self.model_to_train()

            batch = next(self.trainLoader)
            batch = [x.to(dist.get_rank(), non_blocking=True) if type(x) == torch.Tensor else x for x in batch]
            self.task.before_training_step()
            # forward
            output = self.task.training_step(batch)
            self.train_outputs.append(output)
            # backward
            if self.task.automatic_optimization:
                self.task.amp_scaler.scale(output['loss'] / self.gradient_accumulate_every).backward()
                if self.step%self.gradient_accumulate_every == 0:
                    self.task.amp_scaler.unscale_(self.task.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.task.parameters(), 5)
                    self.task.amp_scaler.step(self.task.optimizer)
                    self.task.amp_scaler.update()
                    self.task.optimizer.zero_grad()
            self._lr_update(self.task.scheduler)
            self._progress.on_training_end(output['loss'])

            if self.step != 0 and self.step % self.save_ckpt == 0:
                self.save(milestone=self.step, save_nums = self.save_nums)

            # end training
            # self._progress.on_train_batch_end(output)
            # self.log(self.task.logs)
            # self.reset_log()

            self.task.training_step_end(self.train_outputs)
            self.log(self.task.logs)
            self.reset_log('epoch')
            # self._lr_update(self.task.scheduler, state = 'epoch')

            if self.step != 0 and self.step % self.validation_step == 0:
                # if dist.get_rank() == 0:
                self.model_to_eval()
                with torch.no_grad():
                    self._progress.on_validation_start()
                    for batch_idx, batch in enumerate(self.valLoader):

                        batch = [x.to(dist.get_rank()) if type(x) == torch.Tensor else x for x in batch]
                        output = self.task.validation_step(batch)
                        self.val_outputs.append(output)

                        self._progress.on_validation_batch_end(output)
                        self.log(self.task.logs)
                        self.reset_log('step')
                    self._progress.val_bar_reset()
                    self.task.validation_loop_end(self.val_outputs)
                self.log(self.task.logs)
                if self.ckpt is not None:
                    self.ckpt.monitoring(self.task, self.step)
                self.reset_log('epoch')

                # self._progress.on_epoch_end()

        # if self.ckpt is not None:
        #         self.ckpt.save_ckpt(self.task, self.step, end = True)


#==========================================================

    def _lr_update(self, scheduler):
        self.step += 1
        self.task.milestone = self.step
        if self.task.automatic_scheduler:
            self.lr = self.get_lr(self.task.optimizer)
        else:
            self.lr = self.task.lr

        if scheduler is not None:
            scheduler.step()

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def reset_log(self, level):
        if level == 'epoch':
            self.task.logs = {}
            self.train_outputs = []
            self.val_outputs = []
            self.eval_outputs = []

        elif level == 'step':
            self.task.logs = {}

    def model_to_train(self):
        for module in self.task.children():
            module = module.train()

    def model_to_eval(self):
        for module in self.task.children():
            module = module.eval()

    def save(self, milestone, save_nums = 3):
        data = self.task.state_dicts_list(milestone)

        if dist.get_rank() == 0:
            if os.path.isfile(str(self.results_folder / f'task-{milestone-self.save_ckpt*save_nums}.pt')):
                os.remove(str(self.results_folder / f'task-{milestone-self.save_ckpt*save_nums}.pt'))
        if self.task.args.fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.task, StateDictType.FULL_STATE_DICT, save_policy):
                model_states = self.task.state_dicts_list(milestone)
            if dist.get_rank() == 0:
                torch.save(model_states, str(self.results_folder / f'task-{milestone}.pt'))
        else:
            if dist.get_rank() == 0:
                model_states = self.task.state_dicts_list(milestone)
                torch.save(model_states, str(self.results_folder / f'task-{milestone}.pt'))

    def load(self, log_id, milestone):
        results_folder = os.path.join('./checkpoint', log_id)
        results_folder = Path(results_folder)

        model_states = torch.load(str(results_folder / f'task-{milestone}.pt'), map_location = 'cuda:{}'.format(dist.get_rank()))
        # self.task.load_from_state_dicts(torch.load('/home/gook/Local/E2ETTS/diffe2e_dis/checkpoint/anneal_dis/task-390000.pt', map_location = 'cpu'))
        # model_states = torch.load(str(results_folder / f'task-{milestone}.pt'), map_location = 'cuda:{}'.format(self.device_for_inf))
        self.task.load_from_state_dicts(model_states)
        self.step = model_states['step']
        self._progress = ProgressBar(self.task, self.train_num_steps, initial = self.step)

    # ================ model summary ===============
    def summary(self):
        self._log_device_info(dist.get_rank())
        self._model_info()

    def _log_device_info(self, device):
        if device in ['cpu', 'CPU']:
            pass
        elif type(device) != int:
            raise Warning('type of device should be integer for cude:{}')

        _device_type = 'gpu' if device not in ['cpu', 'CPU'] else 'cpu'
        # _status = 'cpu' if device in ['cpu', 'CPU'] else torch.cuda.current_device()
        _status = 'cpu' if device in ['cpu', 'CPU'] else dist.get_rank()

        print(f"\nGPU available: {torch.cuda.is_available()}, used: {_device_type == 'gpu'}")
        print('LOCAL_RANK: {} - CUDA_VISIBLE_DEVICES: {}'.format(_status, [x for x in range(torch.cuda.device_count())]))
        if device not in ['cpu', 'CPU']:
            print('Device information : {}\n'.format(torch.cuda.get_device_name(device)))

    def _model_info(self):
        summary = OrderedDict((name, module) for name, module in list(self.task.named_children()))
        summary_value = [layer.__class__.__name__ for layer in summary.values()]

        len_name = max(len(layer) for layer in summary.keys())
        len_mod = max(len(layer.__class__.__name__) for layer in summary.values())

        col_widths = [2, len_name+3, len_mod+3, 8]
        cols = [[' ', []], ['Name', []], ['Type', []], ['#Params', []]]
        total_width = sum(col_widths) + 3*len(col_widths)
        n_rows = len(summary)

        for idx, key in enumerate(summary):
            name = key
            type = summary[key].__class__.__name__
            params = self.get_size(summary[key])

            cols[0][1].append(idx)
            cols[1][1].append(name)
            cols[2][1].append(type)
            cols[3][1].append(params)

        s = "{:<{}}"
        header = [s.format(c[0], l) for c, l in zip(cols, col_widths)]
        sums = " | ".join(header) + "\n" + "-" * total_width
        for i in range(n_rows):
            line = []
            for c, l in zip(cols, col_widths):
                line.append(s.format(str(c[1][i]), l))
            sums += "\n" + " | ".join(line)
        sums += "\n" + "-" * total_width
        print(sums)


    def get_size(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn

        return self.byteup(pp)

    def byteup(self, fsize):
        units = ['', ' K', ' M', ' T']
        for idx in range(0, 4):
            if fsize >= 1000:
                fsize /= 1000
            else:
                break
        return str(round(fsize, 2))+units[idx]


# ================ Logger ===============
    def log(self, data):
        if data:
            if self.logger is not None:
                data['step'] = self.step
                self.logger.log(data)

    def save_log(self, data):
        for key in data.keys():
            if key not in self.log_history.keys():
                self.log_history[key] = []
            self.log_history[key].append(data[key])

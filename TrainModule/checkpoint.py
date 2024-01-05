import torch
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

class Monitor():
    '''
    monitor: checkpoint를 저장하는 기준 -> str
        dict format의 self.log의 key값을 받음


    '''

    def __init__(self,
                save_top_k = 3,
                monitor = None,
                mode = 'max',
                ckpt = '.'):

        self.save_top_k = save_top_k
        self.monitor = monitor
        self.ckpt_path = ckpt
        self.mode = mode

        if self.mode in ['max', 'Max']:
            self.filter = min
        elif self.mode in ['min', 'Min']:
            self.filter = max

        if self.monitor is None:
            self.top_k_list = []
        else:
            self.top_k_list = {}
        self.count = 0


    def monitoring(self, Task, step):
        if self.monitor is None:
            self.top_k_list.append(step)
            if self.count < self.save_top_k:
                self.save_ckpt(Task, step)
            else:
                self.delete_ckpt(Task, self.top_k_list.pop(0))
                self.save_ckpt(Task, step)

        elif self.monitor not in Task.logs.keys():
            raise Warning('{} is not in Task log`s key'.format(self.monitor))
        else:
            if Task.logs[self.monitor] not in self.top_k_list.keys():
                if self.count < self.save_top_k:
                    self.top_k_list[Task.logs[self.monitor]] = [step]
                    self.save_ckpt(Task, step) #Task.model
                else:
                    if Task.logs[self.monitor] < self.filter(self.top_k_list):          # fileter가 바뀌면 부등호 방향도 바뀌어야하지 않나
                        pass
                    else:
                        # print('새로운 acc')
                        # print('delete step: {}, save step: {}'.format(self.top_k_list[self.filter(self.top_k_list)][0], step))
                        self.delete_ckpt(Task, self.top_k_list[self.filter(self.top_k_list)].pop(0))
                        if len(self.top_k_list[self.filter(self.top_k_list)]) == 0:
                            del self.top_k_list[self.filter(self.top_k_list)]
                        self.save_ckpt(Task, step)
                        self.top_k_list[Task.logs[self.monitor]] = [step]

            elif Task.logs[self.monitor] in self.top_k_list.keys():
                if self.count < self.save_top_k:
                    self.top_k_list[Task.logs[self.monitor]].append(step)
                    self.save_ckpt(Task, step) #Task.model
                else:
                    if Task.logs[self.monitor] < self.filter(self.top_k_list):
                        pass
                    else:
                        del_step = self.top_k_list[self.filter(self.top_k_list)].pop(0)
                        # print('기존에 있는 acc')
                        # print('delete step: {}, save step: {}'.format(del_step, step))
                        self.delete_ckpt(Task, del_step)
                        self.save_ckpt(Task, step)
                        self.top_k_list[Task.logs[self.monitor]].append(step)

                        if len(self.top_k_list[self.filter(self.top_k_list)]) == 0:
                            del self.top_k_list[self.filter(self.top_k_list)]
                        # print(self.top_k_list)
            # print('count: ', self.count)
            # print(self.top_k_list)
            # print('min: ', min(self.top_k_list), '\n')

    def save_ckpt(self, Task, step, end = False):
        if end:
            for model in Task.model_dict.keys():
                path = os.path.join(self.ckpt_path, model+'_'+'end.ckpt')
                if Task.args.fsdp:
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(Task.model_dict[model], StateDictType.FULL_STATE_DICT, save_policy):
                        model_state = Task.model_dict[model].state_dict()
                else:
                    model_state = Task.model_dict[model].state_dict()
                torch.save(model_state, path)
        else:
            for model in Task.model_dict.keys():
                path = os.path.join(self.ckpt_path, model+'_'+str(step)+'.ckpt')
                if Task.args.fsdp:
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(Task.model_dict[model], StateDictType.FULL_STATE_DICT, save_policy):
                        model_state = Task.model_dict[model].state_dict()
                else:
                    model_state = Task.model_dict[model].state_dict()
                torch.save(model_state, path)
            self.count += 1


    def delete_ckpt(self, Task, step):
        for model in Task.model_dict.keys():
            path = os.path.join(self.ckpt_path, model+'_'+str(step)+'.ckpt')
            if os.path.exists(path):
                os.remove(path)
        self.count -= 1

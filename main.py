## torchrun --nproc_per_node=4 main.py -a true -test true
import warnings
warnings.filterwarnings('ignore')

import yaml
with open('./config.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=int, default = 0)
parser.add_argument('--amp', '-a', type=lambda s: s.lower() in ['true'], default='false')
parser.add_argument('--id', '-id', type=str, default = 'None')
parser.add_argument('--checkpoint', '-ckpt', type=int, default = 0)
parser.add_argument('--inference', '-inference', type=lambda s: s.lower() in ['true'], default='false')
parser.add_argument('--fsdp', '-fsdp', type=lambda s: s.lower() in ['true'], default='true')
parser.add_argument('--test', '-test', type=lambda s: s.lower() in ['true'], default='false')

args = parser.parse_args()

import os

import random
import numpy as np
import torch

import wandb
from TrainModule.Trainer import Trainer
from TrainModule.checkpoint import Monitor
from TrainModule.backup import file_backup
from Task import Task

import torch.distributed as dist

random.seed(conf["system"]["seed"])
np.random.seed(conf["system"]["seed"])
torch.random.manual_seed(conf["system"]["seed"])
torch.manual_seed(conf["system"]["seed"])
torch.cuda.manual_seed(conf["system"]["seed"])

availables = [x for x in range(torch.cuda.device_count())]
dist.init_process_group("nccl")
if args.test:
    logger = None
elif dist.get_rank() == availables[0]:
    if args.id == 'None':
        logger = wandb.init(
                        name = conf['log']['name'],
                        project = conf['log']['project'],
                        config = conf,
                        )
    else:
        logger = wandb.init(
                        id = args.id,
                        name = conf['log']['name'],
                        project = conf['log']['project'],
                        config = conf,
                        resume = True,
                        )

    code_path = os.path.join(logger.dir, "backup")
    backup_dir, conf['system']['monitor']['ckpt'] = file_backup(
                                    prtpath = '.',
                                    run_file_dir = code_path,
                                    ckpt_dir_name = 'best_ckpt'
                                    )
else:
    logger = None

task = Task(args)

trainer = Trainer(
                task = task,
                train_num_steps = conf['training']['train_num_steps'],
                device = args.device,
                monitor = conf['system']['monitor'],
                logger = logger,
                validation_step = conf['system']['validation_step'],
                save_ckpt = conf['system']['save_ckpt']
                )


if args.id != 'None':
    trainer.load(args.id, args.checkpoint)


# dist.init_process_group("nccl")
# rank = dist.get_rank()
# print(f"Start running basic DDP example on rank {rank}.")

# # create model and move it to GPU with id rank
# device_id = rank % torch.cuda.device_count()


trainer.run()

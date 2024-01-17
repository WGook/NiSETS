
import sys
import warnings
warnings.filterwarnings('ignore')

import os
import yaml
from env import AttrDict
import json
with open('./config.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

import torch
from glob import glob

# import wandb
from model.gradtts import E2ETTS

from data_util.dset import LJSpeech
from torch.utils.data import DataLoader

import torchaudio
from tqdm import tqdm

device = 'cuda:0'
e2etts = E2ETTS(**conf['e2etts']).to(device)
e2etts.load_state_dict(torch.load('checkpoint/off_169435906/e2etts_420000.ckpt', map_location= device))

e2etts.eval()
dset = LJSpeech(segment_size=0, valid = False, metafile = conf['dataset']['metafile_valid'])
vl = DataLoader(
                dset,
                batch_size=1,
                shuffle=False,
                collate_fn=dset.collate_fn
                )

with torch.no_grad():
    gamma = 1.02
    temperature = 5
    n_timesteps = 1
    if not os.path.exists('./samples_NiSETS'):
        os.makedirs('./samples_NiSETS')
    for i, batch in tqdm(enumerate(vl)):
        wavs, mel_lens, mels, phones, phone_lens, durations, ids, segment_idcs = batch
        phones = phones.to(device)
        phone_lens = phone_lens.to(device)
        wavs_g_hat, zt = e2etts.sample(phones, phone_lens, spk = None, n_timesteps = n_timesteps, gamma = gamma, stoc = False, prior = True, length_scale = 1., temperature=temperature)

        torchaudio.save('./samples_NiSETS/{}.wav'.format(ids[0]), wavs_g_hat[0].to('cpu').detach(), 22050)
        audio_lens = wavs_g_hat.shape[-1]/22050

from TrainModule.Experiment import ExperimentSetup

import matplotlib.pyplot as plt
import torch
from torch.cuda import amp
from model.gradtts import E2ETTS
from model.hifigan import MultiPeriodDiscriminator, feature_loss, generator_loss, discriminator_loss

import wandb
import yaml
import os

from data_util.dset import LJSpeech
from torch.utils.data import DataLoader

#=======
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#=======
from env import AttrDict
import torch.nn.functional as F
from data_util.meldataset import spectral_normalize_torch, mel_spectrogram
from model.commons import clip_grad_value_, duration_loss, mle_loss, contrastive

with open('./config.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

class Task(ExperimentSetup):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.inference:
            device = args.device
        else:
            device = dist.get_rank()
        self.automatic_optimization = False
        self.automatic_scheduler = False

        torch.cuda.set_device(device)
        
        if not args.inference:
            e2etts = E2ETTS(**conf['e2etts']).cuda(dist.get_rank())
            mpd = MultiPeriodDiscriminator().cuda(dist.get_rank())
            self.e2etts = DDP(e2etts, device_ids=[dist.get_rank()], find_unused_parameters=True)
            self.mpd = DDP(mpd, device_ids=[dist.get_rank()], find_unused_parameters=True)
        else:
            self.e2etts = E2ETTS(**conf['e2etts'])
            self.mpd = MultiPeriodDiscriminator()

        self.model_dict = {'e2etts': self.e2etts.module, 'discriminator': self.mpd.module}
        self.configure_optimizers()

    ###################################################
    ##################  Dataloader  ###################
    ###################################################
    def train_dataloader(self):
        dset = LJSpeech(prtpath = conf['dataset']['prtpath'], metafile = conf['dataset']['metafile_train'])
        sampler = DistributedSampler(dset)
        dl = DataLoader(
            dset,
            batch_size=conf['training']['batch_size'],
            collate_fn = dset.collate_fn,
            sampler = sampler,
            pin_memory=True,
            num_workers = conf['training']['num_workers'],
            )

        
        return self.cycle(dl)

    def val_dataloader(self):
        dset = LJSpeech(segment_size=0, valid = True, metafile = conf['dataset']['metafile_valid'])
        vl = DataLoader(
                        dset,
                        batch_size=conf['training']['batch_size_val'],
                        shuffle=True,
                        collate_fn = dset.collate_fn
                        )
        return vl

    ###################################################
    ##################  Optimizer  ####################
    ###################################################
    def configure_optimizers(self):
        self.optim_g = torch.optim.Adam(self.e2etts.parameters(), conf['training']['learning_rate1'])
        self.optim_d = torch.optim.AdamW(self.mpd.parameters(), conf['training']['learning_rate2'], betas=[conf['training']['beta1'], conf['training']['beta2']], eps = 1e-9, weight_decay = conf['training']['weight_decay'])

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=conf['training']['lr_decay1'], last_epoch=-1)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=conf['training']['lr_decay2'], last_epoch=-1)
        self.amp_scaler = amp.GradScaler(enabled = self.args.amp)

    ###################################################
    ################  Training step  ##################
    ###################################################
    def get_lr(self):
        return self.optim_d.param_groups[-1]['lr']

    def get_mel(self, wav):
        return mel_spectrogram(wav, **conf['feats']).squeeze(0)

    def training_step(self, batch):
        #torch.autograd.set_detect_anomaly(True)
        wavs, mel_lens, mels, phones, phone_lens, durations, ids, segment_idcs = batch
        
        mels = mels.squeeze(1)
        # DDPM
        # self.optim_ddpm.zero_grad()
        with amp.autocast(enabled = self.args.amp):
            (wavs_g_hat, seg_wavs, xeg_mels, seg_mels, seg_Z0s_tilde, seg_Zts), out_ddpm_loss, out_dur_loss, out_prior_loss, tstep = self.e2etts(phones, phone_lens, mels, mel_lens, wavs, segment_idcs = segment_idcs, spk=None, out_size=None)

            seg_g_hat_mel = self.get_mel(wavs_g_hat.squeeze(1))

            wavs_df_hat_r, wavs_df_hat_g, _, _ = self.mpd(seg_wavs, wavs_g_hat.detach(), tstep)
            with amp.autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(wavs_df_hat_r, wavs_df_hat_g)
                loss_d = loss_disc
        self.optim_d.zero_grad()
        self.amp_scaler.scale(loss_d).backward()
        self.amp_scaler.step(self.optim_d)

        with amp.autocast(enabled = self.args.amp):
            wavs_df_hat_r, wavs_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(seg_wavs, wavs_g_hat, tstep)
            with amp.autocast(enabled=False):
                loss_mel = F.l1_loss(xeg_mels, seg_g_hat_mel, reduction = 'none')
                loss_mel = loss_mel.mean()
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_gen_f, losses_gen_f = generator_loss(wavs_df_hat_g)

                loss_ddpm = out_ddpm_loss
                loss_dur = duration_loss(*out_dur_loss, phone_lens)
                loss_prior = out_prior_loss

                weight1 = 1
                weight2 = 1
                loss_gen_all = weight1*(loss_gen_f + loss_fm_f + 45*loss_mel) + weight2*(45*loss_ddpm + loss_dur + 10*loss_prior)

        self.optim_g.zero_grad()
        self.amp_scaler.scale(loss_gen_all).backward()
        self.amp_scaler.step(self.optim_g)
        self.amp_scaler.update()

        self.scheduler_g.step()
        self.scheduler_d.step()

        ############  Log  ############
        self.log({'Loss Generator': loss_gen_all, 'Loss Discriminator': loss_d})

        return {'loss': (loss_gen_all, loss_d, loss_ddpm)}

    def validation_step(self, batch):
        wavs, mel_lens, mels, phones, phone_lens, durations, ids, segment_idcs = batch
        samples, Zt = self.e2etts.module.sample(phones, phone_lens, spk = None, n_timesteps = 1, stoc = False)
        vocoded = self.e2etts.module.generator(mels)

        return {'ground': wavs, 'synth': samples, 'vocoded': vocoded, 'Zt': Zt.squeeze()}

    def validation_loop_end(self, outputs):
        nums = len(outputs)
        fig, ax = plt.subplots(figsize = (30, 6))
        for i, output in enumerate(outputs):
            plt.subplot(1, 4, 1)
            plt.title('Ground')
            plt.pcolor(self.get_mel(output['ground']).to('cpu'))
            plt.colorbar()
            plt.subplot(1, 4, 2)
            plt.title('Vocoder')
            plt.pcolor(self.get_mel(output['vocoded']).to('cpu'))
            plt.colorbar()
            plt.subplot(1, 4, 3)
            plt.title('Synthesized')
            plt.pcolor(self.get_mel(output['synth']).to('cpu'))
            plt.colorbar()
            plt.subplot(1, 4, 4)
            plt.title('Z0')
            plt.pcolor(output['Zt'].to('cpu'))
            plt.colorbar()

        self.log({"Mel": wandb.Image(fig)})
        plt.savefig(os.path.join(self.results_folder, '{}.png'.format(self.milestone)))
        plt.close()


    def state_dicts_list(self, step):
        model_states = {
            'step': step,
            'task': self.state_dict(),
            'optim_g':self.optim_g.state_dict(),
            'optim_d':self.optim_d.state_dict(),
            'sch_g':self.scheduler_g.state_dict(),
            'sch_d':self.scheduler_d.state_dict(),
            'scaler': self.amp_scaler.state_dict()
        }
        return model_states

    def load_from_state_dicts(self, model_states):
        self.load_state_dict(model_states['task'], strict = False)
        self.optim_g.load_state_dict(model_states['optim_g'])
        self.optim_d.load_state_dict(model_states['optim_d'])
        self.scheduler_g.load_state_dict(model_states['sch_g'])
        self.scheduler_d.load_state_dict(model_states['sch_d'])
        self.amp_scaler.load_state_dict(model_states['scaler'])

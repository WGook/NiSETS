import math
import random

import torch
import torch.nn.functional as F

from model import monotonic_align
import numpy as np
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.commons import sequence_mask, generate_path, fix_len_compatibility
from model.hifigan import Generator
from model.unet import UNet
from data_util3.text.symbols import symbols

nsymbols = len(symbols)+1+1 # padding & inter.

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class E2ETTS(BaseModule):
    def __init__(self, h_TextEnc, h_diffusion, h_Denoise_fn, h_Generator, n_spks = 1):
        super(E2ETTS, self).__init__()
        self.n_spks = n_spks
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, 64)
        self.encoder = TextEncoder(n_vocab = nsymbols, **h_TextEnc)
        self.denoise_fn = UNet(**h_Denoise_fn)
        self.decoder = Diffusion(self.denoise_fn, **h_diffusion)
        self.generator = Generator(**h_Generator)

    @torch.no_grad()
    def sample(self, text, text_lens, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0, gamma= 1, prior = False):
        text, text_lens = self.relocate_input([text, text_lens])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `text_m` and log-scaled token durations `log_dur_pred`
        text_m, log_dur_pred, text_mask = self.encoder(text, text_lens, spk)

        w = torch.exp(log_dur_pred) * text_mask
        w_ceil = torch.ceil(w) * length_scale
        mel_lens = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(mel_lens.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(mel_lens, y_max_length_).unsqueeze(1).to(text_mask.dtype)
        attn_mask = text_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get seg_zms
        seg_zms = torch.matmul(attn.squeeze(1).transpose(1, 2), text_m.transpose(1, 2))
        seg_zms = seg_zms.transpose(1, 2)
        encoder_outputs = seg_zms[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(seg_zms, I)
        z = seg_zms + torch.randn_like(seg_zms, device=seg_zms.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, seg_zms, n_timesteps, stoc, spk, gamma, prior = prior)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        wav = self.generator(decoder_outputs)
        'return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]'
        return wav, decoder_outputs#, x0s, timesteps

    def forward(self, text, text_lens, mels, mel_lens, wavs, segment_idcs = None, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            text (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            text_lens (torch.Tensor): lengths of texts in batch.
            mels (torch.Tensor): batch of corresponding mel-spectrograms.
            mel_lens (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `text_m` and log-scaled token durations `log_dur_pred`
        text_m, log_dur_pred, text_mask = self.encoder(text, text_lens, spk)

        z_max_length = mels.size(-1)
        z_mask = sequence_mask(mel_lens, z_max_length).unsqueeze(1).to(text_mask)
        text_logs = torch.zeros_like(text_m)    # std of prior
        attn_mask = text_mask.unsqueeze(-1) * z_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            x_s_sq_r = torch.exp(-2 * text_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - text_logs, [1]).unsqueeze(-1) # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (mels ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((text_m * x_s_sq_r).transpose(1,2), mels) # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (text_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

            attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # Aligned text dependent prior, ZT = z_m
        z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), text_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), text_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        log_dur_mas = torch.log(1e-8 + torch.sum(attn, -1)) * text_mask

        mel_mask = sequence_mask(mel_lens, mels.size(-1)).unsqueeze(1).to(text_mask.dtype)
        with torch.no_grad():
            seg_zms = []
            seg_masks = []
            seg_wavs_ = []
            seg_mels_ = []
            seg_attns_ = []
            for i in range(mels.shape[0]):
                seg_zms.append(z_m[i][:, segment_idcs[i]:segment_idcs[i]+192])
                seg_masks.append(mel_mask[i][:, segment_idcs[i]:segment_idcs[i]+192])
                seg_wavs_.append(wavs[i][:, 256*segment_idcs[i]:256*segment_idcs[i]+192*256])
                seg_mels_.append(mels[i][:, segment_idcs[i]:segment_idcs[i]+192])
                seg_attns_.append(attn.squeeze(1)[i][:, segment_idcs[i]:segment_idcs[i]+192])
            seg_zms = torch.stack(seg_zms, dim = 0)
            seg_masks = torch.stack(seg_masks, dim = 0)
            seg_wavs = torch.stack(seg_wavs_, dim = 0)
            seg_mels = torch.stack(seg_mels_, dim = 0)

        '''
        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * text_mask
        dur_loss = duration_loss(log_dur_pred, logw_, text_lens)
        '''

        # Compute loss of score-based decoder
        (out_ddpm_loss, est_noise, seg_Zts, seg_Z0s_tilde), tstep = self.decoder.compute_loss(seg_mels, seg_masks, seg_zms, spk)

        # Forward of Generator
        with torch.no_grad():
            i = random.randint(0, 192-32)
            seg_wavs = seg_wavs[:, :, 256*i:256*i+8192]
            xeg_mels = seg_mels[:, :, i:i+32]
            features = seg_Z0s_tilde[:, :, i:i+32]


        wavs_g_hat = self.generator(features)

        # Loss sets
        out_dur_loss = (log_dur_pred, log_dur_mas)
        prior_loss = torch.sum(0.5 * ((mels - z_m) ** 2 + math.log(2 * math.pi)) * mel_mask)
        prior_loss = prior_loss / (torch.sum(mel_mask) * 80)

        return (wavs_g_hat, seg_wavs, xeg_mels, seg_mels, seg_Z0s_tilde, seg_Zts), out_ddpm_loss, out_dur_loss, prior_loss, tstep

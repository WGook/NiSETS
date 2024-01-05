from .meldataset import mel_spectrogram
from .text import cleaned_text_to_sequence
from .text.symbols import symbols

import os
import numpy as np
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import random
import h5py

class LJSpeech(Dataset):
    def __init__(self, prtpath = '/home/gook/Datasets/LJSpeech-1.1', metafile = 'train_.csv', sort=False, drop_last=False, valid = False, segment_size = 0):
        self.prtpath = os.path.join(prtpath, 'wavs')
        metapath = os.path.join(prtpath, 'meta', metafile)
        self.metadata = np.loadtxt(metapath, dtype = str, delimiter='\t')
        self.h5path = '/home/gook/Datasets/LJSpeech-1.1/trainp.hdf5'
        self.segment_size = segment_size
        self.cleaners = ["english_cleaners2"]
        self.add_blank = True
        self.sort = sort
        self.drop_last = drop_last
        self.valid = valid
        self.Mel = torchaudio.transforms.MelSpectrogram(sample_rate = 22050,
                                        n_fft = 1024,
                                        hop_length = 256,
                                        n_mels = 80,
                                        onesided=True,
                                        center=False,
                                        normalized=False
                                        )
        self.ds = self.get_dataset()
    def __len__(self):
        if self.valid:
            return 1
        else:
            return len(self.metadata)

    def get_dataset(self):
        f = h5py.File(self.h5path, 'r')
        return f

    def __getitem__(self, idx):
        basename, _, _, _ = self.metadata[idx]
        wav = torch.Tensor(self.ds['wavs'][basename])
        phonemes = self.ds['texts'][basename][0].decode()
        wav = wav.mean(0, keepdims = True) # wav.shape: batch, 1, samples

        text_norm = cleaned_text_to_sequence(phonemes)
        if self.add_blank:
            text_norm = self.intersperse(text_norm, len(symbols)+1)
        text_norm = torch.LongTensor(text_norm)

        duration = np.array([0])
        
        if wav.shape[-1]//256 - 192 < 0:
            segment_idx = 0
        else:
            segment_idx = random.randint(0, wav.shape[-1]//256 - 192)

        sample = {
            "id": basename,
            "phone": text_norm,
            "duration": duration,
            "wav": wav,
            "path": os.path.join(self.prtpath, basename+'.wav'),
            "segment_idx": segment_idx
        }

        return sample

    def segment_wav(self, wav):
        if self.segment_size != 0:
            if wav.size(1) >= self.segment_size:
                max_audio_start = wav.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                wav = wav[:, audio_start:audio_start+self.segment_size]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.segment_size - wav.size(1)), 'constant')
        else:
            wav = wav.squeeze(0)
        return wav

    def get_mel(self, wav):
        return mel_spectrogram(wav.unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000).squeeze(0)

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        phones = [data[idx]["phone"] for idx in idxs]
        wavs = [self.segment_wav(data[idx]["wav"]) for idx in idxs]
        mels = [self.get_mel(w) for w in wavs]
        durations = [data[idx]["duration"] for idx in idxs]
        segment_idcs = [data[idx]["segment_idx"] for idx in idxs] ####

        phone_lens = torch.LongTensor(np.array([phone.shape[0] for phone in phones]))
        wav_lens = torch.LongTensor(np.array([wav.shape[-1] for wav in wavs]))
        mel_lens = torch.LongTensor(np.array([mel.shape[-1] for mel in mels]))

        phones = torch.LongTensor(self.pad_1D(phones, PAD = len(symbols))) #### PAD = 68
        if self.segment_size == 0:
            wavs = torch.Tensor(self.pad_1D(wavs)).unsqueeze(1)
            mels = torch.Tensor(self.pad_2D(mels))
        else:
            wavs = torch.stack(wavs, dim = 0)
            mels = torch.stack(mels, dim = 0)
        durations = torch.LongTensor(self.pad_1D(durations))
        segment_idcs = torch.LongTensor(segment_idcs)

        return wavs, mel_lens, mels, phones, phone_lens, durations, ids, segment_idcs


    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % data_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % data_size)]
        idx_arr = idx_arr.reshape((-1, data_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output[0]


    def pad_1D(self, inputs, PAD=0):
        def pad_data(x, length, PAD):
            x_padded = np.pad(
                x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
            )
            return x_padded

        max_len = max((len(x) for x in inputs))
        padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

        return padded
    
    def pad_2D(self, inputs, maxlen=None):
        def pad(x, max_len, PAD):
            # PAD = x.min()
            if np.shape(x)[1] > max_len:
                raise ValueError("not max_len")

            s = np.shape(x)[-1]
            x_padded = F.pad(x, (0, max_len-s), mode="constant", value=PAD)
            return x_padded

        if maxlen:
            output = np.stack([pad(x, maxlen, x.min()) for x in inputs])
        else:
            max_len = max(np.shape(x)[1] for x in inputs)
            output = np.stack([pad(x, max_len, x.min()) for x in inputs])

        return output
    
    
    def intersperse(self, lst, item):
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result


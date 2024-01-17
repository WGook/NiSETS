# ADVERSARIAL LEARNING ON COMPRESSED POSTERIOR SPACE FOR NON-ITERATIVE SCORE-BASED END-TO-END TEXT-TO-SPEECH

Official implementation of the non-iterative score based E2E TTS (NiSETS) model.
Conference: ICASSP 2024
Authors: Won-Gook Choi, Donghyun Seong, Joon-Hyuk Chang

## Abstract
Score-based generative models have shown the real-like quality of synthesized speech in the text-to-speech (TTS) area.
However, the critical artifact of score-based models is the requirement of a high computational cost due to the iterative sampling algorithm, and it also makes it difficult to fine-tune the score-based TTS-optimized vocoder.
In this study, we propose a method of joint training the score-based TTS model and HiFi-GAN using the compressed log-mel features, and it guarantees a significant speech quality even on the non-iterative sampling.
As a result, the proposed method overcomes some digital artifacts of the synthesized audios compared to the non-iterative sampling of Grad-TTS.
Also, the non-iterative sampling can generate speech faster than other end-to-end TTS models with fewer parameters.

Demo page: Will be updated soon
Multi-speaker: Will be updated soon

## Usage
### 1. Installation
* Install all Python package requirements (**Python==3.9.12)
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

* Build monotonic alignment
<pre>
<code>
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
</code>
</pre>

### 2. Training
1. Modify configuiration in `config.yaml`
    > Ex.

    `system.validation_step`: step interval of save model ckpt & spectrograms

    `system.save_ckpt`: step interval of save whole project (requires for resuming training)

    > If you use `wandb`...

    `log.project`: Prjoect name
    `log.name`: Log name

    > Dataset path

    `dataset.prtpath`: Directory of LJSpeech dataset

    `dataset.metafile_train`: Directory of metadata for training (It's provided in `./data_util/train.txt`)

    `dataset.metafile_valid`: Directory of metadata for training (It's provided in `./data_util/valid.txt`)

2. Training
* From scratch

<pre>
<code>
bash run.sh
</code>
</pre>
All the checkpoints will be saved in `checkpoint`.

You can modify `CUDA_VISIBLE_DEVICES` and `--nproc-Per_node` in `run.sh` if you use multi-GPU.

If you want to log in `wandb`, then set `-test false` in `run.sh`.
If `-test true`, then the folder name of the chekcpoints will be `off_XXXXXXXX`

If you want to train using `amp`, set`-a true` in `run.sh`, but not recommended.


* Resume from checkpoint

Add configurations, `-id folder_name -ckpt number` in `run.sh`.
For example, `-id off_12345678 -ckpt 500000`


3. Inference
<pre>
<code>
python sampling.py
</code>
</pre>

Before sampling, change the `ckpt_path` and `device` in `run.sh`
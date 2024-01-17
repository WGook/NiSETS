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

## Usage


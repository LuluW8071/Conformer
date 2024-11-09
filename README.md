# Conformer: Convolution-augmented Transformer for Speech Recognition

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/Conformer) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Conformer) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Conformer) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Conformer) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Conformer) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Conformer)

</div>

This repository contains an implementation of the paper __Conformer: Convolution-augmented Transformer for Speech Recognition__ using __Lightning AI :zap:__. 

## 📜 Paper & Blogs Review 

- [x] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [x] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100)
- [x] [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)
- [x] [KenLM](https://kheafield.com/code/kenlm/)
- [x] [Boosting Sequence Generation Performance with Beam Search Language Model Decoding](https://towardsdatascience.com/boosting-your-sequence-generation-performance-with-beam-search-language-model-decoding-74ee64de435a)

## Model
![Conformer](https://github.com/LuluW8071/ASR-with-Speech-Sentiment-and-Text-Summarizer/blob/main/docs/conformer.png)

### Model Architecture Params

| Model           | Conformer (S) | Conformer (M) | Conformer (L) |
|-----------------|---------------|---------------|---------------|
| Encoder Layers  | 16            | 16            | 17            |
| Encoder Dim     | 144           | 256           | 512           |
| Attention Heads | 4             | 4             | 8             |
| Conv Kernel Size| 32            | 32            | 32            |
| Decoder Layers  | 1             | 1             | 1             |
| Decoder Dim     | 320           | 640           | 640           |


## Installation

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/LuluW8071/Conformer.git
   cd Conformer
   ```

2. Install **[Pytorch](https://pytorch.org/)** and  required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have `PyTorch` and `Lightning AI` installed.

## Usage

### Training

>[!IMPORTANT]
> Before training make sure you have placed __comet ml api key__ and __project name__ in the environment variable file `.env`.

To train the __Conformer__ model, use the following command for default training configs:

```bash
python3 train.py
```

Customize the pytorch training parameters by passing arguments in `train.py` to suit your needs:

Refer to the provided table to change hyperparameters and train configurations.
| Args                   | Description                                                           | Default Value      |
|------------------------|-----------------------------------------------------------------------|--------------------|
| `-g, --gpus`           | Number of GPUs per node                                               | 1  |
| `-g, --num_workers`           | Number of CPU workers                                               | 8  |
| `-db, --dist_backend`           | Distributed backend to use for training                             | ddp_find_unused_parameters_true  |
| `--epochs`             | Number of total epochs to run                                         | 50                 |
| `--batch_size`         | Size of the batch                                                     | 32                |
| `-lr, --learning_rate`      | Learning rate                                                         | 2e-4  (0.0002)      | 
| `--checkpoint_path` | Checkpoint path to resume training from                                 | None |
| `--precision`        | Precision of the training                                              | 16-mixed |


```bash
python3 train.py 
-g 4                   # Number of GPUs per node for parallel gpu training
-w 8                   # Number of CPU workers for parallel data loading
--epochs 10            # Number of total epochs to run
--batch_size 64        # Size of the batch
-lr 2e-5               # Learning rate
--precision 16-mixed   # Precision of the training
--checkpoint_path path_to_checkpoint.ckpt    # Checkpoint path to resume training from
```

## Results

The model was trained on __Common Voice Dataset 7.0__ and __my personal recordings__ (~ 1200 hours) and validated on splitted dataset (~ 100 hours).

<!-- | Dataset       | WER  |
|---------------|------|
| LibriSpeech   | 5.3% | -->

## Citations

```bibtex
@misc{gulati2020conformerconvolutionaugmentedtransformerspeech,
      title={Conformer: Convolution-augmented Transformer for Speech Recognition}, 
      author={Anmol Gulati and James Qin and Chung-Cheng Chiu and Niki Parmar and Yu Zhang and Jiahui Yu and Wei Han and Shibo Wang and Zhengdong Zhang and Yonghui Wu and Ruoming Pang},
      year={2020},
      url={https://arxiv.org/abs/2005.08100}, 
}
```


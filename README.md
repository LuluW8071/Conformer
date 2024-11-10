# Conformer: Convolution-augmented Transformer for Speech Recognition

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-yellow.svg) ![License](https://img.shields.io/github/license/LuluW8071/Conformer) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Conformer) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Conformer) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Conformer)

</div>

This repository contains an implementation of the paper __Conformer: Convolution-augmented Transformer for Speech Recognition__ with the training scripts supporting training for distributed parallel gpu nodes using __Lightning AI :zap:__.

## 📜 Paper & Blogs Review 

- [x] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [x] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100)
- [x] [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)
- [x] [KenLM](https://kheafield.com/code/kenlm/)
- [x] [Boosting Sequence Generation Performance with Beam Search Language Model Decoding](https://towardsdatascience.com/boosting-your-sequence-generation-performance-with-beam-search-language-model-decoding-74ee64de435a)

## Conformer Model

![Conformer](assets/conformer.png)

### Model Architecture Params

| Model           | Conformer (Small) | Conformer (Medium) | Conformer (Large) |
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

2. Install Required Dependencies

> [!IMPORTANT]  
> Before installing dependencies from `requirements.txt`, make sure you have installed \
>  __No need to install **CUDA ToolKit** and **PyTorch CUDA** for inferencing. But make sure to install **PyTorch CPU**.__
> - [**CUDA ToolKit v11.8/12.1**](https://developer.nvidia.com/cuda-toolkit-archive)
> - [**PyTorch**](https://pytorch.org/)
> - [**SOX**](https://sourceforge.net/projects/sox/)
>     - **For Linux:**
>         ```bash
>         sudo apt update
>         sudo apt install sox libsox-fmt-all build-essential zlib1g-dev libbz2-dev liblzma-dev
>         ```
> 
> - [**PyAudio**](https://people.csail.mit.edu/hubert/pyaudio/)
>     - **For Linux:**
>       ```bash
>       sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
>       sudo apt-get install ffmpeg libav-tools
>       sudo pip install pyaudio    
>       ```

```bash
pip install -r requirements.txt
```

## Usage

### Audio Conversion

> [!NOTE]
> `--not-convert` if you don't want audio conversion.

```bash
py common_voice.py --file_path file_path/to/validated.tsv --save_json_path file_path/to/save/json -w 4 --percent 10 --output_format wav/flac
```

### Training

>[!IMPORTANT]
> Before training make sure you have placed __comet ml api key__ and __project name__ in the environment variable file `.env`.

To train the __Conformer__ model, use the following command:

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

## Experiment Results

The __confomer model (small)__ was trained on __Mozilla Common Voice Dataset 7.0__, __my personal recordings__ and __LibriSpeech Train Set__ (~ 1200 hrs and 960 hrs) and validated on splitted dataset and __LibriSpeech Test Set__ (~ 100 hrs and 10.5 hrs).

#### Train Configuration

| Parameter                        | Value                       |
|-----------------------------------|-----------------------------|
| **GPU Device**                    | 1 L4                        |
| **Batch Size**                    | 16                          |
| **Epochs**                        | 26                          |
| **Optimizer**                     | AdamW                       |
| **Learning Rate (lr)**            | 1e-4                        |
| **Scheduler**                     | Cosine Annealing with Warmup Restart |
| **Min Learning Rate**             | 3e-5                        |

#### Loss Curves 

![Loss Curve](assets/train_loss,val_loss%20VS%20step.jpeg)

#### Metric Evaluation

| Dataset    | Train Loss | Validation Loss | Greedy WER  | Link |
|---------------|----|----|----|---|
|  <img src="https://dagshub.com/repo-avatars/2561" width="30px" /> | 0.4484 | 0.3119 |22.94% | [:link:](https://drive.google.com/uc?id=1XcouMWSncUeNBvGZednuWYK1jdfKisCr)
| <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1rPYa2Q9zPtwLUeZJP3pWeNwmJjRpcLlpdQ&s" width="30px" />   | ![Status](https://img.shields.io/badge/status-in_progress-yellow.svg) |![Status](https://img.shields.io/badge/status-in_progress-yellow.svg) | ![Status](https://img.shields.io/badge/status-in_progress-yellow.svg)

> _Expected WER for CTC+KEN-LM to be __<15% WER__ and inference with CTC+KEN-LM is found to be in the [notebook](https://github.com/LuluW8071/Conformer/blob/main/notebooks/Conformer_Inference_With_CTC_Decoder.ipynb)._


## Citations

```bibtex
@misc{gulati2020conformerconvolutionaugmentedtransformerspeech,
      title={Conformer: Convolution-augmented Transformer for Speech Recognition}, 
      author={Anmol Gulati and James Qin and Chung-Cheng Chiu and Niki Parmar and Yu Zhang and Jiahui Yu and Wei Han and Shibo Wang and Zhengdong Zhang and Yonghui Wu and Ruoming Pang},
      year={2020},
      url={https://arxiv.org/abs/2005.08100}, 
}
```

# A-TLADMM-Net
Accelerated Unfolded Linearized ADMM Networks Inspired by Differential Equations.

## Getting Started
This repository will contain:
1. Synthetic Data `\Synthetic` directory  for the synthetic data experiment;
1. `\Denoising` directory for the image denoising;
1. `\Inpainting` directory for the image inpainting;
1. `\Natural Image CS` directory  for natural image compressive sensing;
1. `\Speech CS` directory for speech data compressive sensing;
1. `\MRI CS` directory for Magnetic Resonance Imaging compressive sensing.

## Prerequisites

The prerequisites are detailed in 'requirements.txt'.

## Implement



### Image Denoising

To train and test our TLADMM and A-TLADMM algorithms on image denoising , please use the following command:

```python
python A-TLADMM.py
```

This performs training with 20 layers A-TLADMM network.



### Image Inpainting

We demonstrate the use of TLADMM and A-TLADMM on natural image inpainting. We show a clear advantage of TLADMM and A-TLADMM versus well-known methods.

In `\Inpainting\saved_models_inpainting`, there exists trained TLADMM and A-TLADMM models. 

To train a new model with 20 layers, simply run the following:

```
python main.py -c3 -tstart 20 -tstep 1 -tend 21
```

- `c3` is the image inpainting scenario
- `tstart` is the initial number of layers
- `tstep` is the increase in layers during training
- `tend` is the final number of layers

To evaluate TLADMM or A-TLADMM on set11, run the following:

```
python eval.py
```



### Speech Data Compressive Sensing

There are codes of training A-(T)ELADMM-Net on two speech datasets. We refer to https://github.com/vicky-k-19/ADMM-DAD

#### How to train A-TLADMM-Net and A-ELADMM-Net on SpeechCommands 

##### Download dataset

Set `DOWNLOAD_DATA=True` in `measure_speech.py` or create a folder `data` in your working directory, then extract `speech_commands_v0.02.tar.gz` into `data/SpeechCommands/speech_commands_v0.02`.

##### Segment data and obtain measurements

Run 

```
python measure_speech.py --dataset speechcommands --measurement-factor 0.25 --ambient-dim 800 --sample-rate 8000
```

This will create a folder in `data/speechcommands_200_800_8000_orth`.

##### Train A-ELADMM-Net

Run 

```
python A-TLADMM-Net_Audio.py --data_path data/speechcommands_200_800_8000_orth --dataset speechcommands --alg A-ELADMM-Net
```

to train the model at CS ratio=200/800=0.25.

##### Train A-TLADMM-Net

Run 

```
python A-TLADMM-Net_Audio.py --data_path data/speechcommands_200_800_8000_orth --dataset speechcommands --alg A-TLADMM-Net
```

to train the model at CS ratio 0.25.

#### How to train A-TLADMM-Net and A-ELADMM-Net on Timit

##### Download data

Get the tarball from 

```
https://drive.google.com/file/d/1Co7I_sWqQFVl0t39fXnBnAmZhV4E1tcd/view?usp=sharing
```

Then move timit.tgz into the `data` folder and run

```
tar xvf timit.tgz
```

##### Segment, measure and train


Run

```
python measure_speech.py --dataset timit --ambient-dim 800 --sample-rate 8000
```

to segment and measure data. It will create a folder `data/timit_200_800_8000_orth`.


```
python A-TLADMM-Net_Audio.py --data_path data/timit_200_800_8000_orth --dataset timit --alg A-TLADMM-Net
```

to train the A-TLADMM-Net at CS ratio 0.25.


#### Extract Spectrograms

The function

```
def save_spectrogram(wav_path):
	...
```

is used to extract spectrograms, and results are saved into folder `results_speech_admm`.




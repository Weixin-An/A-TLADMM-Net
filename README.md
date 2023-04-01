# A-TLADMM-Net
Accelerated Unfolded Linearized ADMM Networks Inspired by Differential Equations.

## Citations
Please cite this conference paper in your publications if this code helps your research:

```
@inproceedings{an2022numerical,
  title={A Numerical DEs Perspective on Unfolded Linearized ADMM Networks for Inverse Problems},
  author={An, Weixin and Yue, Yingjie and Liu, Yuanyuan and Shang, Fanhua and Liu, Hongying},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={5065--5073},
  year={2022}
}
```

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

### Synthetic Data

This part will be completed soon.

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

### Natural Image Compressive Sensing

This part will be completed soon.

### Speech Data Compressive Sensing

This part will be completed soon.

### MRI Compressive Sensing

This part will be completed soon.


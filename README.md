![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/building-pass-green.svg)
[![LICENSE](https://img.shields.io/github/license/JoeyBling/hexo-theme-yilia-plus "LICENSE")](./LICENSE "LICENSE")
# Architecture

```
├── cnn
  ├── README.md                              // Introduction of CNN model.
  ├── src
  │   ├──dataset.py                          // Dataset loader to feed into model.
  │   ├──cnn_for_train.py                    // CNN train model architecture.
  │   ├──cnn.py                              // CNN architecture.
  |   ├──config.py                           // Parser arguments
  ├── utils
  │   ├──log.py                              // log for all the lines in terminal.
  │   ├──plot.py                             // plot loss and accuracy graph.
  ├── scripts
  │   ├──exe.sh                              // shell script for training with our model.
  │   ├──exe_alex.sh                         // shell script for training with Alexnet(Baseline).
  │   ├──exe_res.sh                          // shell script for training with Resnet34(Best Result).
  ├── train.py                               // Train API entry.
```

# Train CIFAR10 with Torch

This is a very simple framework that can try our provided model or test your own PyTorch model on CIFAR10. 

It’s easy to use and flexible at the same time.

## Requirement

Our experiments are conducted under these environments:

| Operating System |                 Device                 |    Torch     |
| :--------------: | :------------------------------------: | :----------: |
|   Ubuntu 20.04   |  NVIDIA GeForce RTX 3090*1, Cuda:11.5  | 1.11.0+cu113 |
|   Ubuntu 20.04   | NVIDIA GeForce RTX 2080Ti*1, Cuda:11.3 | 1.10.0+cu113 |
|                  |                                        |              |

You can first alter the configuration such as the device and some hyperparameters in the **config.py**.

We highly recommend you to use the bash scripts to simply run

```shell
cd [Your Path]/cnn
sh scripts/exe.sh
```

or you can run by 

```
cd [Your Path]/cnn
python train.py
```

The Model will be stored in the checkpoint folder if the testing accuracy is bigger than the threshold accuracy(default: 80) which you can define in the config.py.



We offer 3 types of bash scripts for you to try, which the hyper-parameters are tuned and the result is the same in our report.

# Benchmark

|                         Model(mode)                          | Test Acc |
| :----------------------------------------------------------: | :------: |
|                           Ours(0)                            |  84.950  |
|   [LeNet](https://ieeexplore.ieee.org/document/726791)(1)    |  72.800  |
| [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)(2) |  78.090  |
| [GoogleNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)(3) |  84.120  |
|    [Resnet18](https://arxiv.org/pdf/1512.03385v1.pdf)(4)     |  83.810  |
| [Resent34](https://arxiv.org/pdf/1512.03385v1.pdf)(w pretrian)(5) |  96.620  |
| [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)(w pretrian)(6) |  86.810  |

Configuration of our simple CNN model will be discusses in the report.



**Resnet34** are finetuned with 35 epochs with learning rate 1e-5.

Other models are running under default setting in config.py.

All the result is in folder **appendix** to see our full result.

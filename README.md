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
  │   ├──plot.py                              // plot loss and accuracy graph.
  ├── scripts
  │   ├──exe.sh                              // shell script for training.
  ├── train.py                               // Train API entry.
```

# Train CIFAR10 with Torch

## Requirement

Our experiments are conducted under these environments:

| Operating System |                Device                |    Torch     |
| :--------------: | :----------------------------------: | :----------: |
|   Ubuntu 20.04   | NVIDIA GeForce RTX 3090*1, Cuda:11.5 | 1.11.0+cu113 |
|                  |                                      |              |
|                  |                                      |              |

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


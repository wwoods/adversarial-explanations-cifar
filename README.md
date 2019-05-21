# Example Code for "Reliable Classification Explanations via Adversarial Attacks on Robust Networks"

This code demonstrates the techniques from the above paper, which will be linked at the time it is published.

## Installation

From scratch without a Python environment, installation takes 10-20 minutes.  With Python already installed, installation takes only a few minutes.

Install [PyTorch](https://pytorch.org), `torchvision`, and `click`, potentially via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3:

```bash
$ conda install -c pytorch pytorch torchvision
$ pip install click
```

Code was tested with:

* Python 3.6
* PyTorch 1.1 + torchvision 0.2.2
* `click` 7.0

Any operating system supporting the above libraries should work; we tested using Ubuntu 18.04.

An NVIDIA GPU is not required, but one or more GPUs will greatly accelerate network training.

## Usage

The application has two modes: explaining a trained model, and training a model from scratch.  Additionally, the repository contains several pre-built networks.

The CIFAR-10 dataset will be automatically downloaded via the `torchvision` library; the desired download location for the CIFAR-10 data must be specified via the environment variable `CIFAR10_PATH`.

### Explain

To generate explanations on the first 10 CIFAR-10 testing examples, use the `explain` command.  For example, to use a pre-built network with both adversarial training and the robustness additions from the paper:

```bash
$ python main.py explain prebuilt/resnet44-all.pt [--eps 0.1]
```

This will create images in the `output/` folder, designed to be viewed in alphabetical order.  For example, `output/0-cat` will contain `_input.png`, the unmodified input image; `_real_was_xxx.png`, an explanation using `g_{explain+}` from the paper on the real class (cat); `_second_dog_was_xxx.png`, an explanation using `g_{explain+}` on the most confident class that was not the correct class; and `0_airplane_was_xxx.png`, `1_automobile_was_xxx.png`, `2_bird_was_xxx.png`, ..., `9_truck_was_xxx.png`, an explanation targeted at each class of CIFAR-10 as indicated in the filename.  In all cases, the `_xxx` preceding the `.png` extension indicates the post-softmax confidence of that class on the original image.  The images look like this:

|&nbsp;|&nbsp;|&nbsp;|&nbsp;|&nbsp;|&nbsp;|&nbsp;|&nbsp;|&nbsp;|&nbsp;|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \_input |![Input image](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/_input.png) | \_real |![Real target](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/_real.png) | \_second |![Second target](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/_second.png) | | |
| 0\_airplane |![Airplane](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/0_airplane.png) | 1\_automobile |![Automobile](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/1_automobile.png) | 2\_bird |![Bird](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/2_bird.png) | 3\_cat |![Cat](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/3_cat.png) | 4\_deer |![Deer](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/4_deer.png) |
| 5\_dog |![Dog](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/5_dog.png) | 6\_frog |![Frog](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/6_frog.png) | 7\_horse |![Horse](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/7_horse.png) | 8\_ship |![Ship](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/8_ship.png) | 9\_truck |![Truck](https://github.com/wwoods/adversarial-explanations-cifar/raw/master/example_output/9_truck.png) |

Note that arguments in `[brackets]` are optional.  `--eps X` specifies that the adversarial explanations should be built with `rho=X`.  The process could be further optimized, but presently takes a minute or two.


### Train

To train a new network:

```bash
$ python main.py train path/to/model.pt [--adversarial-training] [--robust-additions] [--l2-min]
```

See `python main.py train --help` for additional information on these options.

Training time varies greatly based on available GPU(s).  with both adversarial training and the robustness additions from the paper, training can take up to several days on a single computer.  Turning off adversarial training leads to a significant speedup, and turning off robustness additions produces a smaller speedup.

At the top of the `main.py` file are many `CAPITAL_CASE` variables which may be modified to affect the training process.  Their definitions match those in the paper.

### Prebuilt Networks

The repository contains four prebuilt networks:

1. `prebuilt/resnet44-standard.pt`: A standard ResNet-44 with no special training.
2. `prebuilt/resnet44-adv-train.pt`: A ResNet-44 trained with `--adversarial-training`.
3. `prebuilt/resnet44-all.pt`: A ResNet-44 trained with `--robust-additions`, `--adversarial-training`, and `--l2-min`.
4. `prebuilt/resnet44-robust.pt`: A ResNet-44 trained with `--robust-additions`.

These correspond with, but are not the same as, the networks denoted N1, N2, N3, and N4 in the paper.


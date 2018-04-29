# Bootstrapping loss function implementation
based on "Training Deep Neural Networks on Noisy Labels with Bootstrapping"
[https://arxiv.org/abs/1412.6596](https://arxiv.org/abs/1412.6596)


## Experiments on MNIST

I try to reproduce paper experiments on MNIST:
```bash
cd examples && python3 mnist_with_tensorboardx.py --mode hard_bootstrap --noise_fraction=0.45
cd examples && python3 mnist_with_tensorboardx.py --mode soft_bootstrap --noise_fraction=0.45
cd examples && python3 mnist_with_tensorboardx.py --mode xentropy --noise_fraction=0.45
```

![img](examples/experiments.png)

### Requirements:

- pytorch 0.3.1
- torchvision
- [ignite](https://github.com/pytorch/ignite)


# Bootstrapping loss function implementation
based on "Training Deep Neural Networks on Noisy Labels with Bootstrapping"
[https://arxiv.org/abs/1412.6596](https://arxiv.org/abs/1412.6596)


## Experiments on MNIST

Experiments on MNIST:
```bash
cd examples/mnist && python main.py run --mode hard_bootstrap --noise_fraction=0.45
cd examples/mnist && python main.py run --mode soft_bootstrap --noise_fraction=0.45
cd examples/mnist && python main.py run --mode xentropy --noise_fraction=0.45
```

```
cd examples/mnist && sh run_experiments.sh >> out 2> log
```

- [Experiments on TRAINS](https://app.ignite.trains.allegro.ai/projects/276a39e824794d1093ecddd8b2afb8d0)

### Requirements:

- pytorch>=1.3
- torchvision>=0.4.1
- [pytorch-ignite](https://github.com/pytorch/ignite)>=0.4.2
- google fire>=0.3.1

```
pip install -r requirements.txt
```
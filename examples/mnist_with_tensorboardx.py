"""
 MNIST example with training and validation monitoring using TensorboardX and Tensorboard.
 Requirements:
    TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python mnist_with_tensorboardx.py --log_dir=/tmp/tensorboard_logs
    ```
"""
from __future__ import print_function
import os
import sys
import random
from datetime import datetime
from argparse import ArgumentParser
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss

sys.path.insert(0, "..")

from bootstrapping_loss import SoftBootstrappingLoss, HardBootstrappingLoss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


label_noise_pattern = {
    # True label -> noise label
    0: 7,
    1: 9,
    2: 0,
    3: 4,
    4: 2,
    5: 1,
    6: 3,
    7: 5,
    8: 6,
    9: 8
}


def noisy_labels(y, a=0.5):

    if random.random() > a:
        return y
    return label_noise_pattern[y]


def get_data_loaders(noise_fraction, train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(download=True, root=".", transform=data_transform,
                          target_transform=partial(noisy_labels, a=noise_fraction),
                          train=True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def create_summary_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def run(mode, noise_fraction, train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir):

    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)

    now = datetime.now()
    log_dir = os.path.join(log_dir, "train_{}_{}__{}".format(mode, noise_fraction, now.strftime("%Y%m%d_%H%M")))
    os.makedirs(log_dir)

    cuda = torch.cuda.is_available()
    train_loader, val_loader = get_data_loaders(noise_fraction, train_batch_size, val_batch_size)

    model = Net()

    writer = create_summary_writer(log_dir)
    if cuda:
        model = model.cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    if mode == 'xentropy':
        criterion = nn.CrossEntropyLoss()
    elif mode == 'soft_bootstrap':
        criterion = SoftBootstrappingLoss(beta=0.95)
    elif mode == 'hard_bootstrap':
        criterion = HardBootstrappingLoss(beta=0.8)
    else:
        raise TypeError("Wrong mode {}, expected: xentropy, soft_bootstrap or hard_bootstrap".format(mode))

    trainer = create_supervised_trainer(model, optimizer, criterion, cuda=cuda)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': CategoricalAccuracy(),
                                                     'nll': Loss(nn.CrossEntropyLoss())},
                                            cuda=cuda)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("valdation/loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser("Training on noisy labels with bootstrapping")
    parser.add_argument('--mode', type=str, choices=['xentropy', 'soft_bootstrap', 'hard_bootstrap'],
                        default='xentropy', help='Loss function mode: cross-entropy or bootstrapping (soft, hard)')
    parser.add_argument('--noise_fraction', type=float, default=0.35,
                        help="Label noise fraction")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=300,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    run(args.mode, args.noise_fraction,
        args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir)

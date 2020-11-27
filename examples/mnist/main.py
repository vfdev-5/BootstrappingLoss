"""
 MNIST example with training and validation monitoring using Tensorboard
 
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python main.py --log_dir=/tmp/tensorboard_logs
    ```
"""
import sys
import random

from pathlib import Path
from datetime import datetime
from functools import partial

import fire

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

import ignite
from ignite.contrib.engines import common
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger


# Add local code
sys.path.insert(0, "../..")

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


def get_data_loaders(data_path, noise_fraction, train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(download=True, root=data_path, transform=data_transform,
                          target_transform=partial(noisy_labels, a=noise_fraction),
                          train=True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4, shuffle=True)

    val_dataset = MNIST(download=False, root=data_path, transform=data_transform, train=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False)
    return train_loader, val_loader


def run(
    data_path="/tmp/MNIST", 
    seed=3321,
    mode="xentropy", 
    noise_fraction=0.35, 
    batch_size=64,
    val_batch_size=1000, 
    num_epochs=50, 
    lr=0.01, 
    momentum=0.5,
    as_pseudo_label=None,
    log_dir="/tmp/output-bootstraping-loss/mnist/",
    with_trains=False,
):
    """Training on noisy labels with bootstrapping

    Args:
        data_path (str): Path to MNIST dataset. Default, "/tmp/MNIST"
        seed (int): Random seed to setup. Default, 3321
        mode (str): Loss function mode: cross-entropy or bootstrapping (soft, hard). 
            Choices 'xentropy', 'soft_bootstrap', 'hard_bootstrap'.
        noise_fraction (float): Label noise fraction. Default, 0.35.
        batch_size (int): Input batch size for training. Default, 64.
        val_batch_size (int): input batch size for validation. Default, 1000.
        num_epochs (int): Number of epochs to train. Default, 50.
        lr (float): Learning rate. Default, 0.01.
        momentum (float): SGD momentum. Default, 0.5.
        log_dir (str): Log directory for Tensorboard log output. Default="/tmp/output-bootstraping-loss/mnist/".
        with_trains (bool): if True, experiment Trains logger is setup. Default, False.

    """
    assert torch.cuda.is_available(), "Training should running on GPU"
    device = "cuda"

    manual_seed(seed)
    logger = setup_logger(name="MNIST-Training")

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Setup output path
    suffix = ""
    if mode == "soft_bootstrap" and (as_pseudo_label is not None and not as_pseudo_label):
        suffix = "as_xreg"
    output_path = Path(log_dir) / "train_{}_{}_{}_{}__{}".format(mode, noise_fraction, suffix, now, num_epochs)

    if not output_path.exists():
        output_path.mkdir(parents=True)    

    parameters = {
        "seed": seed,
        "mode": mode,
        "noise_fraction": noise_fraction,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "momentum": momentum,
        "as_pseudo_label": as_pseudo_label,
    }
    log_basic_info(logger, parameters)

    if with_trains:
        from trains import Task

        task = Task.init("BootstrappingLoss - Experiments on MNIST", task_name=output_path.name)
        # Log hyper parameters
        task.connect(parameters)

    train_loader, test_loader = get_data_loaders(data_path, noise_fraction, batch_size, val_batch_size)
    model = Net().to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    if mode == 'xentropy':
        criterion = nn.CrossEntropyLoss()
    elif mode == 'soft_bootstrap':
        if as_pseudo_label is None:
            as_pseudo_label = True
        criterion = SoftBootstrappingLoss(beta=0.95, as_pseudo_label=as_pseudo_label)
    elif mode == 'hard_bootstrap':
        criterion = HardBootstrappingLoss(beta=0.8)
    else:
        raise ValueError("Wrong mode {}, expected: xentropy, soft_bootstrap or hard_bootstrap".format(mode))

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=True)

    metrics={
        "Accuracy": Accuracy(),
        "{} loss".format(mode): Loss(criterion),
    }
    if mode is not "xentropy":
        metrics["xentropy loss"] = Loss(nn.CrossEntropyLoss())

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, "Train", state.metrics)
        state = evaluator.run(test_loader)
        log_metrics(logger, epoch, "Test", state.metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED | Events.COMPLETED, run_validation)

    evaluators = {"training": train_evaluator, "test": evaluator}
    tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, evaluators=evaluators)

    trainer.run(train_loader, max_epochs=num_epochs)

    test_acc = evaluator.state.metrics["Accuracy"]
    tb_logger.writer.add_hparams(parameters, {"hparam/test_accuracy": test_acc})

    tb_logger.close()

    return (mode, noise_fraction, as_pseudo_label, test_acc)


def log_metrics(logger, epoch, tag, metrics):
    logger.info(
        "Epoch {} - {} metrics: {}".format(
            epoch, tag, " - ".join(["{}: {:.4f}".format(k, v) for k, v in metrics.items()])
        )
    )


def log_basic_info(logger, config):
    logger.info("Train on MNIST")
    logger.info("- PyTorch version: {}".format(torch.__version__))
    logger.info("- Ignite version: {}".format(ignite.__version__))

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info("\t{}: {}".format(key, value))
    logger.info("\n")


if __name__ == "__main__":
    fire.Fire({"run": run})

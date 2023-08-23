

from .dataset_src import *


def get_dataloader(dataset_config):
    if dataset_config.name.lower() == 'cifar10':
        pipeline = CIFAR10Pipeline(dataset_config)
    elif dataset_config.name.lower() == 'cifar100':
        pipeline = CIFAR100Pipeline(dataset_config)
    else:
        raise NotImplementedError
    return pipeline.start()



import torch


def get_optimizer(model, optim_config, T_max=-1, stage=-1):
    """
    :param model: model architecture
    :param optim_config:
    {
        "loss": {
            "name": name of loss function
        },
        "optimizer": {
            "name": name of optimizer
            "lr": learning rate
            "weight_decay": weight decay
            "momentum": momentum
        },
        "scheduler": {
            "name": name of lr scheduler
            "T_max": T_max
        }
    }
    :param T_max: T_max
    :param stage: start from 0
    :return: Torch Optimizer
    """

    if optim_config.loss.name.lower() == 'crossentropyloss':
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        raise NotImplementedError

    if optim_config.optimizer.name.lower() == 'sgd':
        if stage == -1:
            optimizer = torch.optim.SGD(model.parameters(), lr=optim_config.optimizer.lr, weight_decay=optim_config.optimizer.weight_decay, momentum=optim_config.optimizer.momentum)
        else:
            if stage >= len(optim_config.optimizer.weight_decay):
                stage = len(optim_config.optimizer.weight_decay) - 1
            optimizer = torch.optim.SGD(model.parameters(), lr=optim_config.optimizer.lr, weight_decay=optim_config.optimizer.weight_decay[stage], momentum=optim_config.optimizer.momentum)
            print(f'dynamic weight-decay，current： {optim_config.optimizer.weight_decay[stage]}')
    else:
        raise NotImplementedError

    if optim_config.scheduler.name.lower() == 'cosineannealinglr':
        T_max = optim_config.scheduler.T_max if T_max == -1 else T_max
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise NotImplementedError

    return criterion, optimizer, scheduler

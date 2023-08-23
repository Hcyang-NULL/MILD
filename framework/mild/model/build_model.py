

from .from_FINE import *


def get_model(model_config):
    """
    :param model_config: configures of model
    :return: Torch Model
    """
    if model_config.reference == 'FINE':
        if model_config.name.lower() == 'resnet34':
            model = ResNet34(model_config.num_classes)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model = model.cuda()
    return model

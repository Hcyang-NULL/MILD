

import os
import json
from tensorboardX import SummaryWriter


def get_logger(log_config, full_config=None):
    """

    :param full_config:
    :param log_config:
    {
        "log_dir": path for log files
    }
    :return:
    """

    if not os.path.exists(log_config.log_dir):
        os.makedirs(log_config.log_dir)

    log_dir = os.path.join(log_config.log_dir, log_config.title)
    if not os.path.exists(log_dir):
        pass
    else:
        os.system('rm -rf {}'.format(log_dir))

    os.makedirs(log_dir)

    tf_dir = os.path.join(log_dir, 'curve')
    os.makedirs(tf_dir)

    save_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(save_dir)

    if full_config is not None:
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(full_config, f, indent=4)

    writer = SummaryWriter(tf_dir)
    return writer, save_dir

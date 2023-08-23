

import math
import torchvision
from torch.utils.data import Dataset, DataLoader
from aug import *
from framework.mild.noise import *
from reliability.Fitters import Fit_Weibull_Mixture


class CIFAR10Pipeline(object):
    def __init__(self, dataset_config):
        """
        :param dataset_config: configuration of dataset
        """
        self.dataset_config = dataset_config

    def _get_source_data(self):
        train_cifar = torchvision.datasets.CIFAR10(root='./CIFAR', train=True, download=True)
        if self.dataset_config.split_val:
            print('Splitting train/val set')
            train_indices, val_indices = random_train_val_split(train_cifar.targets, is_cifar10=True)
            x_train = train_cifar.data[train_indices]
            y_train = np.array(train_cifar.targets)[train_indices]
            x_val = train_cifar.data[val_indices]
            y_val = np.array(train_cifar.targets)[val_indices]
        else:
            print('No splitting train/val set')
            x_train = train_cifar.data
            y_train = np.array(train_cifar.targets)
            x_val = None
            y_val = None

        test_cifar = torchvision.datasets.CIFAR10(root='./CIFAR', train=False, download=True)
        x_test = test_cifar.data
        y_test = np.array(test_cifar.targets)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_noise(self, y_clean):
        if self.dataset_config.reference == 'FINE':
            if self.dataset_config.noise_type == 'symmetric':
                y_noise = FINENoise.cifar10_sym_noise(y_clean, self.dataset_config.noise_rate)
            elif self.dataset_config.noise_type == 'asymmetric':
                y_noise = FINENoise.cifar10_asym_noise(y_clean, self.dataset_config.noise_rate)
            else:
                raise ValueError('Invalid noise type')
        else:
            raise ValueError('unrecognized noise generation reference')
        return y_noise

    def start(self):
        x_train, y_train_clean, x_val, y_val, x_test, y_test = self._get_source_data()

        y_train_noise = self._generate_noise(y_train_clean)

        data_transforms = get_cifar10_transforms()

        train_dataset = MildDataset(x_train, y_train_noise, y_train_clean, data_transforms.train, self.dataset_config.batch_size)
        if x_val is not None:
            val_dataset = MildDataset(x_val, None, y_val, data_transforms.test)
        else:
            val_dataset = None
        test_dataset = MildDataset(x_test, None, y_test, data_transforms.test)

        train_dataloader = DataLoader(train_dataset, batch_size=self.dataset_config.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=self.dataset_config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        else:
            val_dataloader = None
        test_dataloader = DataLoader(test_dataset, batch_size=self.dataset_config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        return train_dataloader, val_dataloader, test_dataloader, 10


class CIFAR100Pipeline(object):
    def __init__(self, dataset_config):
        """
        :param dataset_config: configuration of dataset
        """
        self.dataset_config = dataset_config

    def _get_source_data(self):
        train_cifar = torchvision.datasets.CIFAR100(root='./CIFAR', train=True, download=True)
        if self.dataset_config.split_val:
            print('Splitting train/val set')
            train_indices, val_indices = random_train_val_split(train_cifar.targets, is_cifar100=True)
            x_train = train_cifar.data[train_indices]
            y_train = np.array(train_cifar.targets)[train_indices]
            x_val = train_cifar.data[val_indices]
            y_val = np.array(train_cifar.targets)[val_indices]
        else:
            print('No splitting train/val set')
            x_train = train_cifar.data
            y_train = np.array(train_cifar.targets)
            x_val = None
            y_val = None

        test_cifar = torchvision.datasets.CIFAR100(root='./CIFAR', train=False, download=True)
        x_test = test_cifar.data
        y_test = np.array(test_cifar.targets)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_noise(self, y_clean):
        if self.dataset_config.reference == 'FINE':
            if self.dataset_config.noise_type == 'symmetric':
                y_noise = FINENoise.cifar100_sym_noise(y_clean, self.dataset_config.noise_rate)
            elif self.dataset_config.noise_type == 'asymmetric':
                y_noise = FINENoise.cifar100_asym_noise(y_clean, self.dataset_config.noise_rate)
            else:
                raise ValueError('Invalid noise type')
        else:
            raise ValueError('unrecognized noise generation reference')
        return y_noise

    def start(self):
        x_train, y_train_clean, x_val, y_val, x_test, y_test = self._get_source_data()

        y_train_noise = self._generate_noise(y_train_clean)

        data_transforms = get_cifar100_transforms()

        train_dataset = MildDataset(x_train, y_train_noise, y_train_clean, data_transforms.train, self.dataset_config.batch_size)
        if x_val is not None:
            val_dataset = MildDataset(x_val, None, y_val, data_transforms.test)
        else:
            val_dataset = None
        test_dataset = MildDataset(x_test, None, y_test, data_transforms.test)

        train_dataloader = DataLoader(train_dataset, batch_size=self.dataset_config.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=self.dataset_config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        else:
            val_dataloader = None
        test_dataloader = DataLoader(test_dataset, batch_size=self.dataset_config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        return train_dataloader, val_dataloader, test_dataloader, 100


class MildDataset(Dataset):
    def __init__(self, x, y_noise, y_clean, data_transform, batch_size=-1, total_clean=-1):
        self.x_ids = [i for i in range(len(x))]
        self.x = x
        self.y_clean = y_clean
        self.y_noise = y_noise
        self.data_transform = data_transform
        self.batch_size = batch_size
        self.total_clean = total_clean

        self.x_seq = {i: [] for i in range(len(x))}
        self._computed()

        self.alpha1s = -1
        self.beta1s = -1
        self.alpha2s = -1
        self.beta2s = -1
        self.proportion1s = -1

    def seq_update(self, x_ids, values):
        for i, x_id in enumerate(x_ids):
            self.x_seq[x_id.item()].append(values[i].item())

    def mild_selection(self, metric_func):
        x_id_metric = [(x_id, metric_func(self.x_seq[x_id])) for x_id in self.x_ids]
        x_id_metric.sort(key=lambda k: k[1])
        x_metric = [item[1] for item in x_id_metric]

        min_metric = min(x_metric)
        if min_metric <= 0:
            for_fit_metric = [item - min_metric for item in x_metric]
        else:
            for_fit_metric = deepcopy(x_metric)
        for_fit_metric = [item if item != 0 else 0.1 for item in for_fit_metric]
        try:
            results = Fit_Weibull_Mixture(failures=for_fit_metric)
        except ValueError:
            print(f'weibull mixture fit failed!')
            return None, None

        if math.isnan(results.alpha_1):
            print(f'weibull mixture fit failed! NaN!')
            return None, None

        pdf = results.distribution.PDF(xmin=0, xmax=100)
        tmp_pdf_x = 0
        for i in range(len(pdf) - 1, -1, -1):
            if pdf[i] > pdf[i - 1]:
                tmp_pdf_x = i
                break

        noise_pdf_x = tmp_pdf_x / float(len(pdf)) * 100

        if noise_pdf_x <= 2 or True:
            noise_pdf_x = max(results.alpha_1, results.alpha_2)

        select_x_ids = [item[0] for item in x_id_metric if item[1] <= noise_pdf_x]

        x = self.x[select_x_ids]
        y_clean = self.y_clean[select_x_ids]
        y_noise = self.y_noise[select_x_ids]
        mild_dataset = MildDataset(x, y_noise, y_clean, self.data_transform, self.batch_size, total_clean=self.total_clean)
        mild_dataloader = DataLoader(mild_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)

        mild_dataset.alpha1s = results.alpha_1
        mild_dataset.alpha2s = results.alpha_2
        mild_dataset.beta1s = results.beta_1
        mild_dataset.beta2s = results.beta_2
        mild_dataset.proportion1s = results.proportion_1

        select_num = len(select_x_ids)
        select_rate = select_num / len(self.x_ids)

        print('Select {} ({}%) samples and clean rate is {} and clean recall is {}'.format(select_num, int(select_rate * 100), round(mild_dataset.clean_rate, 4), round(mild_dataset.recall, 4)))

        return mild_dataloader, noise_pdf_x

    def _computed(self):
        self.clean_rate = np.sum(self.y_clean == self.y_noise) / len(self.y_clean)
        if self.total_clean == -1:
            self.total_clean = np.sum(self.y_clean == self.y_noise)
        self.recall = np.sum(self.y_clean == self.y_noise) / self.total_clean if self.total_clean > 0 else -1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_id = self.x_ids[idx]
        x = self.x[idx]
        y_clean = self.y_clean[idx]
        y_noise = self.y_noise[idx] if self.y_noise is not None else None
        x = self.data_transform(x)

        if y_noise is None:
            return x_id, x, y_clean
        else:
            return x_id, x, y_noise, y_clean


def random_train_val_split(base_dataset, is_cifar10=False, is_cifar100=False):
    if is_cifar100:
        num_classes = 100
    elif is_cifar10:
        num_classes = 10
    else:
        raise ValueError('Invalid dataset')
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

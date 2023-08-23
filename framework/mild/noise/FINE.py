

import numpy as np
from copy import deepcopy
from random import sample, choice
from numpy.testing import assert_array_almost_equal


class FINENoise(object):
    @staticmethod
    def cifar10_sym_noise(y_clean, noise_rate: float):
        assert 0 <= noise_rate <= 1
        
        data_num = len(y_clean)
        y_noise = deepcopy(y_clean)
        label_set = list(set(y_clean.tolist()))
        noisy_indices = sorted(sample([i for i in range(data_num)], int(noise_rate * data_num)))
        
        for noisy_index in noisy_indices:
            y_noise[noisy_index] = choice(label_set)
        actual_noise_rate = (y_clean != y_noise).mean()
        print(f'Actual Noise: {actual_noise_rate}')
        return y_noise

    @staticmethod
    def cifar100_sym_noise(y_clean, noise_rate: float):
        return FINENoise.cifar10_sym_noise(y_clean, noise_rate)

    @staticmethod
    def cifar10_asym_noise(y_clean, noise_rate: float):
        y_noise = deepcopy(y_clean)
        for i in range(10):
            indices = np.where(y_clean == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < noise_rate * len(indices):
                    # @ truck -> automobile
                    if i == 9:
                        y_noise[idx] = 1
                    # @ bird -> airplane
                    elif i == 2:
                        y_noise[idx] = 0
                    # @ cat -> dog
                    elif i == 3:
                        y_noise[idx] = 5
                    # @ dog -> cat
                    elif i == 5:
                        y_noise[idx] = 3
                    # @ deer -> horse
                    elif i == 4:
                        y_noise[idx] = 7
        actual_noise_rate = (y_noise != y_clean).mean()
        print(f'Actual Noise: {actual_noise_rate}')
        return y_noise

    @staticmethod
    def cifar100_asym_noise(y_clean, noise_rate: float):
        y_noise = deepcopy(y_clean)
        P = np.eye(100)
        n = noise_rate
        nb_superclasses = 20
        nb_subclasses = 5
        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = FINE_build_for_cifar100(nb_subclasses, n)

            y_train_noisy = FINE_multiclass_noisify(y_noise, P=P, random_state=0)
            actual_noise = (y_train_noisy != y_clean).mean()
            print(f'Actual Noise: {actual_noise}')
            return y_train_noisy


# Following code is copied from FINE


def FINE_build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise

    # adjust last row
    P[size - 1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def FINE_multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

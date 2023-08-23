
def metric_update(loss, pred, y_noise):
    return (pred == y_noise).int()


def _calc_mild(seq: list):
    tmp = ''.join([str(item) for item in seq])
    tmp1 = [item for item in tmp.split('0') if item]
    p1 = len(''.join(tmp1)) if len(tmp1) != 0 else 0
    tmp2 = [item for item in tmp.split('1') if item]
    p2 = len(''.join(tmp2)) if len(tmp2) != 0 else 0
    return p1, p2


def mild_metric(seq: list):
    p1, p2 = _calc_mild(seq)
    return p2 - p1


def get_sample_selection():
    return metric_update, mild_metric

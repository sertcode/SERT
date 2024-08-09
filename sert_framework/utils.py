import numpy as np


# number -> 0,1,2,3 percentile label
def masked_percentile_label(value, mask=None):
    if mask is not None:
        value = value[mask]
    value_sorted = np.argsort(value)
    num_valid = value.shape[0]
    label = np.zeros(num_valid)
    for i in range(num_valid):
        idx = value_sorted[i]
        if i < int(0.25 * num_valid):
            label[idx] = 0
        elif i < int(0.5 * num_valid):
            label[idx] = 1
        elif i < int(0.75 * num_valid):
            label[idx] = 2
        else:
            label[idx] = 3
    return label


# number -> [0, 1] normalized number
def min_max_normalize(data, percentile=0.999):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl) * percentile)]
    # print('max_val: = ', max_val)
    min_val = max(0, sl[0])
    data[data > max_val] = max_val
    data -= min_val
    data /= (max_val - min_val)
    return data, max_val, min_val


def split_x_y(data, lag, val_num=60 * 24, test_num=60 * 24):
    """
    source: train = 8 month, val = 2 month, test = 2 month
    target: train = 30/7/3 days, val = 2 month, test = 2 month
    """
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    num_samples = int(data.shape[0])
    for i in range(-int(min(lag)), num_samples):
        x_idx = [int(_ + i) for _ in lag]
        y_idx = [i]
        x_ = data[x_idx, :, :]
        y_ = data[y_idx, :, :]
        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_y.append(y_)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_y.append(y_)
        else:
            test_x.append(x_)
            test_y.append(y_)
    return np.stack(train_x, axis=0), np.stack(train_y, axis=0), np.stack(val_x, axis=0), np.stack(val_y, axis=0), \
        np.stack(test_x, axis=0), np.stack(test_y, axis=0)

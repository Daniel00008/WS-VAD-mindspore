import visdom
import mindspore
import mindspore.nn as nn

import numpy as np
import random


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))

    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)

    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)


# divide a video into length segments
def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int)  # np.linspace(起始点，结束点，生成的样本数量)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype=np.uint16)
    return r


def save_best_record(test_info, file_path):
    fo = open(file_path, "a")
    fo.write("\nstep: {}\t".format(test_info["epoch"][-1]))
    fo.write("{}\t".format(test_info["test_AUC"][-1]))
    fo.close()

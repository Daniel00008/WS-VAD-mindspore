import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F

from non_local import Aggregate
import ipdb




def weight_init(m):
    classname = m.__class__.__name__
    if classname == 'GraphConv':
        return
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if m.bias is not None:
            m.bias.data.fill_(0)
            
class Classifier(nn.Cell):
    def __init__(self, n_features):
        super(Classifier, self).__init__()
        self.fc1 = nn.Dense(n_features, 512)
        self.fc2 = nn.Dense(512, 128)
        self.fc3 = nn.Dense(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.sigmoid = nn.Sigmoid()


    def construct(self, features):
        features = self.dropout(features)
        scores = self.relu(self.fc1(features))
        scores = self.dropout(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.dropout(scores)
        scores = self.sigmoid(self.fc3(scores))
        return scores


class attention(nn.Cell):
    def __init__(self, n_feature):
        super().__init__()
        self.attention = nn.SequentialCell(nn.Conv1d(n_feature, 512, 3,pad_mode='pad', padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.7),
                                       nn.Conv1d(512, 512, 3,pad_mode='pad', padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.7),
                                       nn.Sigmoid())

    def construct(self, feat):
        x_atn = self.attention(feat)
        return x_atn


class Model(nn.Cell):
    def __init__(self, cfg, flag):
        super(Model, self).__init__()

        self.flag = flag
        self.bs = cfg.BS
        self.feat_dim = cfg.FEATS_DIM
        self.aggregate = Aggregate(self.feat_dim)
        self.classifier = Classifier(self.feat_dim)
        self.attn = attention(self.feat_dim)
        self.sig = nn.Sigmoid()
        self.drop_out = nn.Dropout(0.7)

    def construct(self, x):
        t, f = x.shape
        n_crops = 1
        bs = 1
        x = x.view(-1, t, f)
        nfeat = self.aggregate(x)
        x_atn = self.attn(nfeat)
        x_cls = self.classifier(nfeat.permute(0, 2, 1))

        if self.flag == 'test':
            x_cls = x_atn.permute(0, 2, 1) * x_cls
            return mindspore.ops.mean(x_cls.reshape(bs, n_crops, -1), axis=1).unsqueeze(2)
        else:
            return  {'feat': nfeat, 'scores': x_cls, 'attn': x_atn.transpose(-1, -2)}

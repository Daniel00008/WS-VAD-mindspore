from model import *
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings

import mindspore
from mindspore.ops import stop_gradient
from mindspore import Tensor, ops
warnings.filterwarnings("ignore")
import ipdb

def test(args, test_loader, net):
    net.flag = "test"
    frame_gt = np.load("list/gt-ucf.npy")
    frame_predict = None

    cls_label = []
    predict_list = []
    for i, data in enumerate(test_loader):
        # ipdb.set_trace()
        _data, _label, _ = data[0], data[1], data[2]

        a_predict = net(_data)
        predict_list.append(a_predict)
        if (i + 1) % 10 == 0:
            temp_predict = ops.cat(predict_list, axis=0)
            cls_label.append(int(_label))
            predict_mean = temp_predict.mean(0).numpy()
            fpre_ = np.repeat(predict_mean, 16)
            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])
            predict_list = []

    fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
    auc_score = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(frame_gt, frame_predict, )
    ap_score = auc(recall, precision)

    print("auc:{} ap:{}".format(auc_score, ap_score))

    return auc_score, ap_score

import pdb
from mindspore.dataset import GeneratorDataset
from config import *
from train import *
from model import *
import os
from dataset import *
import mindspore
import mindspore as ms
from utils import save_best_record, set_seed
from model import Model
from train import train
from test_10crop import test
import option
from config import *

from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor, load_checkpoint

if __name__ == "__main__":
    args = option.parser.parse_args()
    config = Config(args)
    if args.debug:
        pdb.set_trace()

    worker_init_fn = None
    if args.seed >= 0:
        set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)

    model = Model(args, flag='test')
    save_mm_model_pth = './ckpt/come605-i3d-mindspore.ckpt'
    best_model_dict =  load_checkpoint(save_mm_model_pth)
    ms.load_param_into_net(model, best_model_dict)

    test_loader = GeneratorDataset(UCF_crime(mode='Test'), column_names=["data","label","name" ], shuffle=False, num_parallel_workers=1)

    test(args, test_loader, model)


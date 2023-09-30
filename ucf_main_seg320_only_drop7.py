import pdb
import mindspore.dataset.GeneratorDataset as GeneratorDataset
from config import *
from train import *
from model import *
import os
from dataset import *
from tqdm import tqdm
import mindspore
import mindspore as ms
from utils import save_best_record, set_seed
from model import Model
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from config import *


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
    model = model.to(device)

    normal_train_loader = GeneratorDataset(
        UCF_crime(mode='Train', is_normal=True), batch_size=64, shuffle=True, num_workers=0,
        worker_init_fn=worker_init_fn, drop_last=True)
    abnormal_train_loader = GeneratorDataset(
        UCF_crime(mode='Train', is_normal=False), batch_size=64, shuffle=True, num_workers=0,
        worker_init_fn=worker_init_fn, drop_last=True)
    test_loader = GeneratorDataset(
        UCF_crime(mode='Test'), batch_size=1, shuffle=False, num_workers=0, worker_init_fn=worker_init_fn)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = './one_stream_smooth_1_ucf_seg320_10sep_only_drop7'
    optimizer = mindspore.nn.Adam(model.parameters(), lr=config.lr[0], betas=(0.9, 0.999), weight_decay=0.00005)

    test(args, test_loader, model, device)


    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)

        train(args, step, normal_loader_iter, abnormal_loader_iter, model, optimizer, device)

        if step % 5 == 0 and step > 10:
            auc, ap = test(args, test_loader, model, device)
            test_info["epoch"].append(step)
            if args.dataset == 'xd':
                test_info["test_AUC"].append(ap)
                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    ms.save_checkpoint(model, './ckpt/' + args.model_name + '{}-i3d.ckpt'.format(step, async_save=True)
                    save_best_record(test_info, os.path.join(output_path, 'results.txt'))
            else:
                test_info["test_AUC"].append(auc)
                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    ms.save_checkpoint(model, './ckpt/' + args.model_name + '{}-i3d.ckpt'.format(step, async_save=True)
                    save_best_record(test_info, os.path.join(output_path, 'results.txt'))



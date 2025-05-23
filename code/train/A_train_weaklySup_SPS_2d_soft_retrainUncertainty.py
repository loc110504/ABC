#!/usr/bin/env python3
import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader.Dataset_all import BaseDataSets
from dataloader.transform_2D import RandomGenerator
from networks.net_factory import net_factory
from val_2D import test_all_case_2D

# --- device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='../../data/ACDC2017/ACDC_for2D', help='Data root path')
parser.add_argument('--data_type', type=str,
                    default='Heart', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='ACDC', help='Data name')  
parser.add_argument('--trainData', type=str,
                    default='trainReT01.txt', help='retrain Data')
parser.add_argument('--validData', type=str,
                    default='valid.txt', help='vaild Data')
                 
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--exp', type=str,
                    default='A_weakly_SPS_2d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage2', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='pseudoLab', help='supervision type')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--ES_interval', type=int,
                    default=10000, help='early-stopping patience')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()


def train(args, snapshot_path):
    # unpack
    data_root_path = args.data_root_path
    batch_size     = args.batch_size
    base_lr        = args.base_lr
    num_classes    = args.num_classes
    max_iterations = args.max_iterations
    trainData_txt  = args.trainData
    validData_txt  = args.validData
    ES_interval    = args.ES_interval

    # model setup
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model.to(device)
    model_parameter = sum(p.numel() for p in model.parameters())
    logging.info("model_parameter:{}M".format(round(model_parameter / (1024*1024),2)))

    # load stage1 weights
    snapshot_path_ori = f"../../model/{args.data_type}_{args.data_name}/{args.exp}_{args.model}_stage1"
    save_mode_path_ori = os.path.join(snapshot_path_ori, f"{args.model}_best_model.pth")
    model.load_state_dict(torch.load(save_mode_path_ori, map_location=device))

    # datasets
    db_train = BaseDataSets(
        base_dir=data_root_path, 
        split="train",
        data_txt=trainData_txt,
        transform=transforms.Compose([
            RandomGenerator(args.patch_size, args.num_classes)
        ]),
        sup_type=args.sup_type,
        num_classes=num_classes
    )
    db_val = BaseDataSets(
        base_dir=data_root_path, 
        split="val",
        data_txt=validData_txt,
        num_classes=num_classes
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val,   batch_size=1,         shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss  = CrossEntropyLoss(ignore_index=num_classes)
    ce_loss2 = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"{len(trainloader)} iterations per epoch")

    iter_num       = 0
    max_epoch      = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    fresh_iter_num = 0

    for epoch_num in range(max_epoch):
        epoch_loss_sum   = 0.0
        epoch_ce_sum     = 0.0
        epoch_pseudo_sum = 0.0

        model.train()
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].to(device)
            label_batch  = sampled_batch['label'].to(device)

            outputs, outputs_aux1 = model(volume_batch)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)

            # CE loss
            loss_ce1 = ce_loss(outputs, label_batch.long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch.long())
            loss_ce  = 0.5 * (loss_ce1 + loss_ce2)

            # soft pseudo-label loss
            alpha = random.random() + 1e-1
            soft_pseudo_label = alpha * outputs_soft1.detach() + (1.0-alpha) * outputs_soft2.detach()
            loss_pse = 0.5 * (ce_loss2(outputs_soft1, soft_pseudo_label) + ce_loss2(outputs_soft2, soft_pseudo_label))

            loss = loss_ce + 8.0 * loss_pse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr decay
            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # tensorboard
            writer.add_scalar('info/lr',           lr,              iter_num)
            writer.add_scalar('info/total_loss',   loss.item(),    iter_num)
            writer.add_scalar('info/loss_ce1',loss_ce1,iter_num)
            writer.add_scalar('info/loss_ce2',loss_ce2,iter_num)
            writer.add_scalar('info/loss_ce',      loss_ce.item(), iter_num)  
            writer.add_scalar('info/loss_pse_sup', loss_pse.item(),iter_num)

            # accumulate
            epoch_loss_sum   += loss.item()
            epoch_ce_sum     += loss_ce.item()
            epoch_pseudo_sum += loss_pse.item()

            iter_num += 1

            # validation & checkpoint
            if iter_num % 200 == 0:
                logging.info(f"iteration {iter_num}: loss_ce={loss_ce.item():.4f}, loss_pse={loss_pse.item():.4f}, alpha={alpha:.3f}")
                model.eval()
                metric_list = test_all_case_2D(valloader, model, args)
                for class_i in range(num_classes-1):
                    writer.add_scalar(f'info/val_{class_i+1}_dice', metric_list[class_i], iter_num)
                mean_dice = metric_list[:,0].mean()
                writer.add_scalar('info/val_dice_score', mean_dice, iter_num)
                if mean_dice > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = mean_dice
                    save_model_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_performance:.4f}.pth')
                    save_best      = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_model_path)
                    torch.save(model.state_dict(), save_best)
                model.train()

            if iter_num >= max_iterations or (iter_num - fresh_iter_num) >= ES_interval:
                break
        # end for batch

        # log average losses this epoch
        n_batches = len(trainloader)
        avg_loss   = epoch_loss_sum   / n_batches
        avg_ce     = epoch_ce_sum     / n_batches
        avg_pse    = epoch_pseudo_sum / n_batches
        logging.info(f"[Epoch {epoch_num+1}/{max_epoch}] avg_loss={avg_loss:.4f}, avg_ce={avg_ce:.4f}, avg_pseudo={avg_pse:.4f}")

        if iter_num >= max_iterations or (iter_num - fresh_iter_num) >= ES_interval:
            break

    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    snapshot_path = f"../../model/{args.data_type}_{args.data_name}/{args.exp}_{args.model}_{args.fold}"
    os.makedirs(snapshot_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(__file__, os.path.join(snapshot_path, f"{run_id}_{os.path.basename(__file__)}"))

    # setup logger
    logger = logging.getLogger()
    logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(snapshot_path, "train_log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    logging.info(str(args))

    start_time = time.time()
    train(args, snapshot_path)
    elapsed = time.time() - start_time
    logging.info(f"time cost: {elapsed:.2f} s, i.e. {elapsed/3600:.2f} h")

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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.Dataset_all import BaseDataSets
from dataloader.transform_2D import RandomGenerator
from networks.net_factory import net_factory
from val_2D import test_all_case_2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/ACDC2017/ACDC_for2D', help='Data root path')
parser.add_argument('--data_type', type=str,
                    default='Heart', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='ACDC', help='Data name')  
parser.add_argument('--trainData', type=str,
                    default='train.txt', help='train Data, select mode: train, \
                        trainrless2, trainrlessd4, trainrlessd8, trainrlessd16')
parser.add_argument('--validData', type=str,
                    default='valid.txt', help='vaild Data')
                 
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name, select: unet_cct, \
                        NestedUNet2d_2dual, swinunet_2dual')
parser.add_argument('--exp', type=str,
                    default='A_weakly_SPS_2d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type:label, scribble or pseudoLab')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--ES_interval', type=int,
                    default=10000, help='maximum iteration iternal for early-stopping')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input. Specially, [224, 224] for swinunet')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()
"""ACDC: total 1356 samples; 30 samples for vadilation;
57 iterations per epoch; max epoch: 527.
"""

def train(args, snapshot_path):
    data_root_path = args.root_path
    batch_size = args.batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    trainData_txt = args.trainData
    validData_txt = args.validData
    ES_interval = args.ES_interval

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model_parameter = sum(p.numel() for p in model.parameters())
    logging.info("model_parameter:{}M".format(round(model_parameter / (1024*1024),2)))

    db_train = BaseDataSets(
        base_dir=data_root_path, 
        split="train",
        data_txt = trainData_txt,
        transform=transforms.Compose([
            RandomGenerator(args.patch_size, args.num_classes)
            ]),  
        sup_type=args.sup_type,
        num_classes=num_classes
        )
    db_val = BaseDataSets(
        base_dir=data_root_path, 
        split="val",
        data_txt = validData_txt,
        num_classes=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    ce_loss2 = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, gt_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['gt']
            volume_batch, label_batch, gt_batch = volume_batch.cuda(), label_batch.cuda(), gt_batch.cuda()

            outputs, outputs_aux1 = model(volume_batch)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)
            
            #pCE
            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)#式3

            #mix soft pseudo label
            alpha = random.random() + 1e-10

            soft_pseudo_label = alpha * outputs_soft1.detach() + (1.0-alpha) * outputs_soft2.detach()

            loss_pse_sup_soft = 0.5*(ce_loss2(outputs_soft1, soft_pseudo_label) +ce_loss2(outputs_soft2, soft_pseudo_label) )

            loss = loss_ce + 8.0 * loss_pse_sup_soft
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce1',loss_ce1,iter_num)
            writer.add_scalar('info/loss_ce2',loss_ce2,iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_sps',loss_pse_sup_soft,iter_num)

            #draw
            # if iter_num % 100 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)

            #     outputs = torch.argmax(torch.softmax(
            #         outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction',
            #                      outputs[1, ...] * 30, iter_num)

            #     labs = label_batch[1, ...].unsqueeze(0) * 30
            #     writer.add_image('train/Label', labs, iter_num)

            #     gt = gt_batch[1, ...].unsqueeze(0) * 30
            #     writer.add_image('train/GroundTruth', gt, iter_num)



            if iter_num > 0 and iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup_soft: %f, alpha: %f' 
                    %(iter_num, loss.item(), loss_ce.item(), loss_pse_sup_soft.item(), alpha))
                
                model.eval()
                metric_list = test_all_case_2D(valloader, model, args)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i], iter_num)
             
                if metric_list[:, 0].mean() > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = metric_list[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', metric_list[:, 0].mean(), iter_num)
                logging.info("avg_metric:{} ".format(metric_list))
                logging.info('iteration %d : dice_score : %f ' % (iter_num, metric_list[:, 0].mean()))

                model.train()


            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num - fresh_iter_num >= ES_interval:
                logging.info("early stooping since there is no model updating over 1w \
                    iteration, iter:{} ".format(iter_num))
                break

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations or (iter_num - fresh_iter_num >= ES_interval):
            iterator.close()
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
    torch.cuda.manual_seed(args.seed)
    
    snapshot_path = "../../model/{}_{}/{}_{}_{}".format(
        args.data_type, args.data_name, args.exp, args.model, args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, os.path.join(snapshot_path, run_id + "_" + os.path.basename(__file__))
    )

    # logging.basicConfig(filename=snapshot_path+"/train_log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logger = logging.getLogger()
    logger.handlers.clear()
    file_handler = logging.FileHandler(snapshot_path+"/train_log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s')) 
    logger.addHandler(console_handler)
    logging.info(str(args))
    start_time = time.time()
    train(args, snapshot_path)
    time_s = time.time()-start_time
    logging.info("time cost: {} s, i.e, {} h".format(time_s,time_s/3600))
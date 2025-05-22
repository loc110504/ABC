import sys
import os
import glob
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import time
import logging
import random
from PIL import Image

from networks.net_factory import net_factory
from utils import (
    calculate_metric_percase, logInference,
    get_rgb_from_uncertainty, get_the_first_k_largest_components
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path',    type=str,
                    default='../../data/ACDC2017/ACDC_for2D',
                    help='Root folder chứa *_volumes và .txt')
parser.add_argument('--testData',          type=str,
                    default='trainvol.txt',
                    help='File danh sách .h5: trainvol.txt/test.txt/valid.txt')
parser.add_argument('--savedir',           type=str,
                    default='TrResult',
                    help='Tên thư mục con kết quả: TrResult/ValResult/TsResult')
parser.add_argument('--model',             type=str,
                    default='unet_cct',
                    help='Tên mô hình')
parser.add_argument('--exp',               type=str,
                    default='A_weakly_SPS_2d',
                    help='Tên experiment')
parser.add_argument('--fold',              type=str,
                    default='stage1',
                    help='Fold name')
parser.add_argument('--num_classes',       type=int,
                    default=4, help='Số lớp segmentation')
parser.add_argument('--tt_num',            type=int,
                    default=4, help='Số lần test (uncertainty)')
parser.add_argument('--threshold',         type=float,
                    default=0.1, help='Ngưỡng uncertainty')
parser.add_argument('--uncertainty_show_path', type=str,
                    default='../../figure/MedIA_ACDC/uncertainty/',
                    help='Folder lưu ảnh minh họa uncertainty')
parser.add_argument('--seed',              type=int,
                    default=2022, help='Random seed')

def weight_with_uncertainty_class(preds, C):
    entropy = -torch.sum(preds * torch.log(preds + 1e-6),
                         dim=1, keepdim=True)
    return entropy / torch.log(torch.tensor(C).cuda())

def get_rgb_from_label_ACDC(label):
    h, w = label.shape
    img = Image.fromarray(label.astype(np.uint8))
    out = Image.new('RGB', (w, h), (0,0,0))
    for i in range(w):
        for j in range(h):
            p = img.getpixel((i,j))
            if   p==0: c=(0,0,0)
            elif p==1: c=(255,0,0)
            elif p==2: c=(0,255,0)
            elif p==3: c=(0,0,255)
            elif p==4: c=(255,255,0)
            out.putpixel((i,j), c)
    return out

def convertMap(itk_map):
    itk_map[itk_map==0] = 5
    itk_map[itk_map==4] = 0
    itk_map[itk_map==5] = 4
    return itk_map

def find_org_nii(case_name, root_for2d):
    base = root_for2d.replace("_for2D", "")
    pattern = os.path.join(base, "images_N", "**", f"{case_name}.nii.gz")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy {case_name}.nii.gz ở {pattern}")
    return matches[0]

def test_single_volume_2d_forTrainUncertainty(case_path, net,
                                             test_save_path, FLAGS,
                                             uncertainty_path_save):
    case_name = os.path.basename(case_path).replace(".h5","")
    logging.info(f"Testing case: {case_name}")

    with h5py.File(case_path,'r') as f:
        image    = f['image'][:]
        label    = f['label'][:]
        scribble = f['scribble'][:]

    Z, H, W = image.shape
    pred_main = np.zeros_like(label, dtype=np.uint8)
    pred_auxi = np.zeros_like(label, dtype=np.uint8)
    pred_mean = np.zeros_like(label, dtype=np.uint8)
    pred_cum  = np.zeros_like(label, dtype=np.uint8)

    for ind in range(Z):
        gt = label[ind]
        get_rgb_from_label_ACDC(gt).save(
            os.path.join(uncertainty_path_save,
                         f"{case_name}_slice{ind}_gt.png"))
        scr = scribble[ind]
        get_rgb_from_label_ACDC(scr).save(
            os.path.join(uncertainty_path_save,
                         f"{case_name}_slice{ind}_scri.png"))

        sl = image[ind]
        sl_resized = zoom(sl, (256/H, 256/W), order=0)
        inp = torch.from_numpy(sl_resized)\
                   .unsqueeze(0).unsqueeze(0)\
                   .float().cuda()

        net.eval()
        with torch.no_grad():
            out1, out2 = net(inp)
            sm1 = torch.softmax(out1, dim=1)
            sm2 = torch.softmax(out2, dim=1)
            smm = (sm1 + sm2) / 2.0

            for arr, sm, tag in ((pred_main, sm1, "pred_main"),
                                 (pred_auxi, sm2, "pred_auxi"),
                                 (pred_mean, smm, "pred_mean")):
                arg = torch.argmax(sm, dim=1).squeeze(0).cpu().numpy()
                pr = zoom(arg, (H/256, W/256), order=0).astype(np.uint8)
                arr[ind] = pr
                get_rgb_from_label_ACDC(pr).save(
                    os.path.join(uncertainty_path_save,
                                 f"{case_name}_slice{ind}_{tag}.png"))

            unc = weight_with_uncertainty_class(smm, FLAGS.num_classes)
            uw = zoom(unc.cpu().squeeze().numpy(),
                      (H/256, W/256), order=0)
            mask = uw < FLAGS.threshold
            get_rgb_from_uncertainty(uw).save(
                os.path.join(uncertainty_path_save,
                             f"{case_name}_slice{ind}_uncertainty.png"))
            Image.fromarray(mask.astype(np.uint8)*255).save(
                os.path.join(uncertainty_path_save,
                             f"{case_name}_slice{ind}_mask.png"))

            # apply mask + convertMap for unannotated
            pu = pred_mean * mask
            pu_map = convertMap(pred_mean.copy()) * mask
            pu_map = convertMap(pu_map)
            get_rgb_from_label_ACDC(pu_map).save(
                os.path.join(uncertainty_path_save,
                             f"{case_name}_slice{ind}_cu_unannotated.png"))

            # largest-component post-processing
            lab_out = np.zeros_like(pr)
            for c in range(1, FLAGS.num_classes):
                comp = get_the_first_k_largest_components(
                    pu==c, 1)
                lab_out = np.where(comp, c, lab_out)
            pred_cum[ind] = lab_out

            lab_map = np.zeros_like(pr)
            for c in range(1, FLAGS.num_classes+1):
                comp = get_the_first_k_largest_components(
                    pu_map==c, 1)
                lab_map = np.where(comp, c, lab_map)
            lab_map = convertMap(lab_map)
            get_rgb_from_label_ACDC(lab_map).save(
                os.path.join(uncertainty_path_save,
                             f"{case_name}_slice{ind}_cuM_unannotated.png"))

    # metadata từ NIfTI gốc
    org_nii = find_org_nii(case_name, FLAGS.data_root_path)
    logging.info(f"Loaded meta from {org_nii}")
    org_itk = sitk.ReadImage(org_nii)
    spacing = org_itk.GetSpacing()

    # tính metric
    m_main, m_auxi, m_cum = [], [], []
    for c in range(1, FLAGS.num_classes):
        m_main.append(
            calculate_metric_percase(pred_main==c, label==c,
                                     (spacing[2],spacing[0],spacing[1])))
        m_auxi.append(
            calculate_metric_percase(pred_auxi==c, label==c,
                                     (spacing[2],spacing[0],spacing[1])))
        m_cum.append(
            calculate_metric_percase(pred_cum==c, label==c,
                                     (spacing[2],spacing[0],spacing[1])))

    # lưu kết quả NIfTI
    def save_itk(arr, tag):
        itk = sitk.GetImageFromArray(arr.astype(np.float32))
        itk.CopyInformation(org_itk)
        sitk.WriteImage(itk,
            os.path.join(test_save_path,
                         f"{case_name}_{tag}.nii.gz"))

    save_itk(pred_main,   "pred1")
    save_itk(pred_auxi,   "pred2")
    save_itk(pred_cum,    "pred_cuM")
    save_itk(image.astype(np.float32), "img")
    save_itk(label.astype(np.float32), "gt")
    save_itk(scribble.astype(np.float32), "scri")

    return [m_main, m_auxi, m_cum]

def Inference(FLAGS, test_save_path):
    # 1) đọc danh sách HDF5
    with open(os.path.join(FLAGS.data_root_path, FLAGS.testData), 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    h5_list = [os.path.join(FLAGS.data_root_path, l) for l in lines]
    logging.info(f"Test volumes: {len(h5_list)}")

    # 2) load model
    snap = os.path.join(
        "../../model",
        f"{FLAGS.data_type}_{FLAGS.data_name}",
        f"{FLAGS.exp}_{FLAGS.model}_{FLAGS.fold}",
        f"{FLAGS.model}_best_model.pth"
    )
    net = net_factory(net_type=FLAGS.model,
                      in_chns=1,
                      class_num=FLAGS.num_classes)
    net.load_state_dict(torch.load(snap))
    net.cuda().eval()
    logging.info(f"Loaded checkpoint: {snap}")

    # 3) ensure uncertainty folder exists
    os.makedirs(FLAGS.uncertainty_show_path, exist_ok=True)

    all_m1, all_m2, all_m3 = [], [], []
    for case in tqdm(h5_list):
        m1, m2, m3 = test_single_volume_2d_forTrainUncertainty(
            case, net, test_save_path,
            FLAGS, FLAGS.uncertainty_show_path
        )
        all_m1.append(np.asarray(m1))
        all_m2.append(np.asarray(m2))
        all_m3.append(np.asarray(m3))

    logging.info("Main predictions:")
    logInference(all_m1)
    logging.info("Auxi predictions:")
    logInference(all_m2)
    logging.info(f"Uncertainty-filtered + post-proc (th={FLAGS.threshold}):")
    logInference(all_m3)

if __name__ == '__main__':
    start = time.time()
    FLAGS = parser.parse_args()

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    # tạo thư mục kết quả + log
    test_save_path = os.path.join(
        "../../result",
        f"{FLAGS.data_type}_{FLAGS.data_name}",
        f"{FLAGS.exp}_{FLAGS.model}_{FLAGS.savedir}_{FLAGS.tt_num}"
    )
    shutil.rmtree(test_save_path, ignore_errors=True)
    os.makedirs(test_save_path + "/log", exist_ok=True)

    # setup logging
    logger = logging.getLogger()
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(test_save_path, "test_info.txt"))
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)

    logging.info("Flags:")
    logging.info(FLAGS)

    Inference(FLAGS, test_save_path)

    elapsed = time.time() - start
    logging.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

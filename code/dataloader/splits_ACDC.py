#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# ==== CẤU HÌNH ==== 
data_root    = Path("../../data/ACDC2017/ACDC") # thư mục gốc chứa images_N/patientXXX
data_2d = Path("../../data/ACDC2017/ACDC_for2D")              
split_txt  = {
    "train":   data_2d/"train.txt",              # danh sách train volumes
    "val":     data_2d/"valid.txt",              # danh sách validation volumes
    "TestSet": data_2d/"test.txt",               # danh sách test volumes
}
imagesN_root = data_root/"images_N"                # chứa các folder patient001/, patient002/, ...
do_move      = False                              # True nếu bạn muốn move thay vì copy

def find_nii(base_name: str) -> Path:
    """
    Tìm file {base_name}.nii.gz trong bất kỳ patientXXX/ nào dưới imagesN_root.
    Ví dụ base_name="patient001_frame01" sẽ tìm:
      images_N/patient001/patient001_frame01.nii.gz
    """
    pattern = f"*/{base_name}.nii.gz"
    matches = list(imagesN_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy {base_name}.nii.gz dưới {imagesN_root}"
        )
    return matches[0]


# ==== Chia splits và copy ảnh + labels ====
for split, txt_path in split_txt.items():
    # Tạo folder đích nếu chưa có
    img_dst = data_root / split / "images_N"
    lbl_dst = data_root / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Đọc danh sách slices .h5
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for h5_rel in lines:
        # Ví dụ h5_rel = "train_slices/patient001_frame01_slice_0.h5"
        # Lấy phần stem: "patient001_frame01_slice_0"
        stem = Path(h5_rel).stem
        # Tách lấy base = "patient001_frame01"
        base = stem.split("_slice_")[0]

        # 1) Copy (hoặc move) file ảnh .nii.gz
        src_img = find_nii(base)
        dst_img = img_dst / src_img.name
        (shutil.move if do_move else shutil.copy2)(src_img, dst_img)

        # 2) Copy (hoặc move) file label .nii.gz (suffix "_gt")
        src_lbl = find_nii(base + "_gt")
        dst_lbl = lbl_dst / src_lbl.name
        (shutil.move if do_move else shutil.copy2)(src_lbl, dst_lbl)

    print(f"[OK] Split '{split}': {len(lines)} entries →")
    print(f"     images in   {img_dst}")
    print(f"     labels in   {lbl_dst}")

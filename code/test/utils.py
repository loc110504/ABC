import numpy as np
from medpy import metric
from scipy import ndimage
import logging
from PIL import Image
# from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast

def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        # asd = asd_fast(pred, gt, voxelspacing=spacing)
        # hd95 = hd95_fast(pred, gt, voxelspacing=spacing)
        return np.array([dice, asd, hd95])
    else:
        return np.array([0,0,0])

def logInference(metric_onTest_all_list):
    metric_onTest_all_list = np.asarray(metric_onTest_all_list)  
    avg_metric_on_list = np.mean(metric_onTest_all_list,axis=0)
    logging.info("avg metric of each organ[dsc,asd,hd95]:{}".format(avg_metric_on_list))
    metricOf3 = np.mean(avg_metric_on_list,axis=0)
    logging.info("mean metric of all[dice,asd,hd95]:{}".format(metricOf3))
    std = np.std(metric_onTest_all_list,axis = 0)
    logging.info("std of each organ[dice,asd,hd95]:{}".format(std))
    std_1= np.mean(std,axis=0)
    logging.info("std of all[dsc,asd,hd95]:{}\n\n".format(std_1))

def get_the_first_k_largest_components_bug(image, k = 1): #image is binary map
    """
    get the largest component from 2D or 3D binary image;
    k refers to the k maximum connected components to be retained; 
    image: nd array;
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sorted = sorted(list(sizes))
    output = np.zeros_like(image, np.uint8)
    for i in range(k):
        max_i_label = np.where(sizes == sizes_sorted[-1-i])[0] + 1 
        output = output + np.asarray(labeled_array == max_i_label, np.uint8)
    return  output

def get_the_first_k_largest_components(image, k=1):
    """
    Giữ lại k thành phần liên thông lớn nhất trong ảnh nhị phân 2D hoặc 3D.
    image: ndarray nhị phân
    k: số thành phần lớn nhất cần giữ
    Trả về mảng uint8 cùng shape với image, chứa các thành phần được chọn.
    """
    if image.sum() == 0:
        # không có thành phần nào
        return image.astype(np.uint8)

    # chọn structuring element phù hợp
    dim = image.ndim
    if dim == 2:
        structure = ndimage.generate_binary_structure(2, 1)
    elif dim == 3:
        structure = ndimage.generate_binary_structure(3, 1)
    else:
        raise ValueError("Dimension phải là 2 hoặc 3")

    # đánh nhãn
    labeled_array, num_features = ndimage.label(image, structure)
    # tính kích thước mỗi nhãn
    sizes = ndimage.sum(image, labeled_array, range(1, num_features + 1))
    # sắp xếp kích thước tăng dần
    sizes_sorted = np.sort(sizes)

    # chuẩn bị output
    output = np.zeros_like(image, dtype=np.uint8)

    # lấy k nhãn lớn nhất
    # nếu có tie (cùng size), np.where sẽ trả về mảng nhiều nhãn, và np.isin xử lý ổn
    largest_sizes = sizes_sorted[-k:]
    # tìm nhãn tương ứng
    labels_to_keep = np.where(np.isin(sizes, largest_sizes))[0] + 1

    # tạo mask những pixel thuộc bất kỳ nhãn nào trong labels_to_keep
    mask = np.isin(labeled_array, labels_to_keep)
    output[mask] = 1

    return output

def get_intersection_region_perclass(image1, image2):
    """get the cross region from two 2D or 3D binary image"""
    intersection = np.logical_and(image1, image2)

    if(intersection.sum() == 0 ):
        print('the intersection is null')
    return intersection

def map_scalar_to_color(x):
    x_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    c_list = [[0, 0, 255],
              [0, 255, 255],
              [0, 255, 0],
              [255, 255, 0],
              [255, 0, 0]]
    for i in range(len(x_list)):
        if(x <= x_list[i + 1]):
            x0 = x_list[i]
            x1 = x_list[i + 1]
            c0 = c_list[i]
            c1 = c_list[i + 1]
            alpha = (x - x0)/(x1 - x0)
            c = [c0[j]*(1 - alpha) + c1[j] * alpha for j in range(3)]
            c = [int(item) for item in c]
            return tuple(c)
        
def get_rgb_from_uncertainty(uncertainty_weight):
    h, w = uncertainty_weight.shape
    uncertainty_weight = (uncertainty_weight - uncertainty_weight.min()) / (uncertainty_weight.max() - uncertainty_weight.min())
    uncertainty_img = Image.fromarray(uncertainty_weight) 
    output_map = Image.new('RGB', tuple((w,h)), (0, 0, 0))
    for i in range(w):
        for j in range(h):
            p0 = uncertainty_img.getpixel((i, j))
            p1 = map_scalar_to_color(p0)
            output_map.putpixel((i,j), p1)
    return output_map
import argparse
import os
import shutil

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory
from configs import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd95, asd


def test_single_volume(case, net):
    h5f = h5py.File(os.path.join(cfg.DATASETS.ROOT_PATH, "data", f"{case}.h5"), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    net.eval()
    with torch.no_grad():
        for ind in range(image.shape[0]):
            slice_img = image[ind, :, :]
            x, y = slice_img.shape[0], slice_img.shape[1]

            slice_res = zoom(slice_img, (256 / x, 256 / y), order=0)

            input_tensor = torch.from_numpy(slice_res).unsqueeze(0).unsqueeze(0).float().cuda()

            out_main = net(input_tensor)
            if isinstance(out_main, (tuple, list)):
                out_main = out_main[0]

            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().numpy()

            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    metrics = []
    for cls in [1, 2, 3]:
        if np.sum(prediction == cls) == 0:
            metrics.append((0.0, 0.0, 128.0, 128.0))
        else:
            metrics.append(calculate_metric_percase(prediction == cls, label == cls))

    return metrics[0], metrics[1], metrics[2]


def Inference():
    with open(os.path.join(cfg.DATASETS.ROOT_PATH, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.strip().split(".")[0] for item in image_list if item.strip()])

    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_ACDC")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"

    snapshot_path = os.path.join(base_save_dir, exp_folder, cfg.TEST.STAGE_NAME)
    test_save_path = os.path.join(base_save_dir, exp_folder, f"{cfg.MODEL.NAME}_predictions/")

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    net = net_factory(net_type=cfg.MODEL.NAME, in_chns=1, class_num=cfg.MODEL.NUM_CLASSES).cuda()
    save_model_path = os.path.join(snapshot_path, f'{cfg.MODEL.NAME}_best_model.pth')

    print(f"Init weight from {save_model_path}")
    checkpoint = torch.load(save_model_path)
    net.load_state_dict(checkpoint['net'] if 'net' in checkpoint else checkpoint)
    net.eval()

    first_total = np.zeros(4)
    second_total = np.zeros(4)
    third_total = np.zeros(4)

    for case in tqdm(image_list, desc="Testing ACDC"):
        m1, m2, m3 = test_single_volume(case, net)
        first_total += np.array(m1)
        second_total += np.array(m2)
        third_total += np.array(m3)

    num_cases = len(image_list)
    avg_metric = [first_total / num_cases, second_total / num_cases, third_total / num_cases]
    return avg_metric, test_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BCP Test ACDC")
    parser.add_argument("-cfg", "--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.opts:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)

    metric_avg, test_save_path = Inference()

    mean_metrics = np.mean(metric_avg, axis=0)

    header = f"{'Dice↑':<10} {'mIoU↑':<10} {'95HD↓':<10} {'ASD↓':<10}"

    print("\n" + "-" * 45)
    print(header)
    for i in range(3):
        print(f"{metric_avg[i][0]:<10.4f} {metric_avg[i][1]:<10.4f} {metric_avg[i][2]:<10.2f} {metric_avg[i][3]:<10.2f}")
    print("-" * 45)
    print(f"{mean_metrics[0]:<10.4f} {mean_metrics[1]:<10.4f} {mean_metrics[2]:<10.2f} {mean_metrics[3]:<10.2f}  (Mean)")
    print("-" * 45 + "\n")

    with open(os.path.join(test_save_path, '../performance.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{header}\n")
        for i in range(3):
            f.write(f"{metric_avg[i][0]:<10.4f} {metric_avg[i][1]:<10.4f} {metric_avg[i][2]:<10.2f} {metric_avg[i][3]:<10.2f}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{mean_metrics[0]:<10.4f} {mean_metrics[1]:<10.4f} {mean_metrics[2]:<10.2f} {mean_metrics[3]:<10.2f}  (Mean)\n")
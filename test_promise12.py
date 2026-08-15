import argparse
import os
import shutil
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm

from networks.net_factory import net_factory
from configs import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, asd


def test_single_volume(case, net):
    npy_data_path = os.path.join(cfg.DATASETS.ROOT_PATH, 'npy_image')
    img = np.load(os.path.join(npy_data_path, f'{case}.npy'))
    mask = np.load(os.path.join(npy_data_path, f'{case}_segmentation.npy'))

    prediction = np.zeros_like(mask)

    net.eval()
    with torch.no_grad():
        for ind in range(img.shape[0]):
            slice_img = img[ind, :, :]
            input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()

            out_main = net(input_tensor)
            if isinstance(out_main, (tuple, list)):
                out_main = out_main[0]

            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            prediction[ind] = out.cpu().numpy()

    if np.sum(prediction == 1) == 0:
        first_metric = 0.0, 0.0
    else:
        first_metric = calculate_metric_percase(prediction == 1, mask == 1)

    return first_metric


def Inference():
    with open(os.path.join(cfg.DATASETS.ROOT_PATH, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.strip().split(".")[0] for item in image_list if item.strip()])

    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_Promise12")
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

    first_total = 0.0
    second_total = 0.0

    for case in tqdm(image_list, desc="Testing Promise12"):
        dice, asd = test_single_volume(case, net)
        first_total += dice
        second_total += asd

    avg_metric = [first_total / len(image_list), second_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BCP Test Promise12")
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

    header = f"{'Dice↑':<10} {'ASD↓':<10}"
    result_str = f"{metric_avg[0]:<10.4f} {metric_avg[1]:<10.2f}  (Mean)"

    print("\n" + "-" * 35)
    print(header)
    print(result_str)
    print("-" * 35 + "\n")

    with open(os.path.join(test_save_path, '../performance.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{header}\n")
        f.write(f"{result_str}\n")
import argparse
import os
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

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt) * 100.0
        hd = metric.binary.hd(pred, gt)
        return dice, hd
    elif pred.sum() == 0 and gt.sum() == 0:
        return 100.0, 0.0
    else:
        return 0.0, 128.0


def test_single_volume(case, net):
    npy_dir = os.path.join(cfg.DATASETS.ROOT_PATH, 'npy_224')
    img_npy_path = os.path.join(npy_dir, f"{case}_img.npy")
    lab_npy_path = os.path.join(npy_dir, f"{case}_lab.npy")

    if not os.path.exists(img_npy_path) or not os.path.exists(lab_npy_path):
        raise FileNotFoundError(f"找不到预处理后的npy缓存文件: {case}，请检查是否运行了训练预处理。")

    image = np.load(img_npy_path)
    label = np.load(lab_npy_path)

    input_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()

    net.eval()
    with torch.no_grad():
        out_main = net(input_tensor)
        if isinstance(out_main, (tuple, list)):
            out_main = out_main[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()

    return calculate_metric_percase(pred == 1, label == 1)


def Inference():
    with open(os.path.join(cfg.DATASETS.ROOT_PATH, 'test.list'), 'r') as f:
        image_list = [item.strip() for item in f.readlines() if item.strip()]

    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_ISIC")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"
    snapshot_path = os.path.join(base_save_dir, exp_folder, cfg.TEST.STAGE_NAME)

    net = net_factory(net_type=cfg.MODEL.NAME, in_chns=3, class_num=cfg.MODEL.NUM_CLASSES).cuda()
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))

    checkpoint = torch.load(save_model_path)
    if 'net' in checkpoint:
        net.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint)

    print("Init weight from {}".format(save_model_path))
    net.eval()

    total_metrics = np.zeros(2)
    for case in tqdm(image_list, desc="Testing ISIC"):
        total_metrics += np.asarray(test_single_volume(case, net))

    avg_metric = total_metrics / len(image_list)
    return avg_metric, os.path.join(base_save_dir, exp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BCP Test ISIC")
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

    metric_vals, save_dir = Inference()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    header = f"{'Dice↑':<10} {'HD↓':<10}"
    result_str = f"{metric_vals[0]:<10.1f} {metric_vals[1]:<10.1f}"

    print("\n" + "-" * 25)
    print(header)
    print(result_str)
    print("-" * 25 + "\n")

    with open(os.path.join(save_dir, 'performance.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{header}\n")
        f.write(f"{result_str}\n")
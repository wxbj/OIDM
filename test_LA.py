import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case
from configs import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_calculate_metric(image_list, snapshot_path, test_save_path):
    model = net_factory(net_type=cfg.MODEL.NAME, in_chns=1, class_num=cfg.MODEL.NUM_CLASSES, mode="test").cuda()
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))

    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['net'] if 'net' in checkpoint else checkpoint)
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=cfg.MODEL.NUM_CLASSES,
                               patch_size=tuple(cfg.INPUT.PATCH_SIZE), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path,
                               metric_detail=cfg.TEST.DETAIL, nms=cfg.TEST.NMS)

    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BCP Test LA")
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

    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_LA")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"

    snapshot_path = os.path.join(base_save_dir, exp_folder, cfg.TEST.STAGE_NAME)
    test_save_path = os.path.join(base_save_dir, exp_folder, f"{cfg.MODEL.NAME}_predictions/")

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    print(test_save_path)

    with open(cfg.DATASETS.ROOT_PATH + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [cfg.DATASETS.ROOT_PATH + "/2018LA_Seg_Training Set/" + item.strip() + "/mri_norm2.h5" for
                  item in image_list if item.strip()]

    metric = test_calculate_metric(image_list, snapshot_path, test_save_path)

    header = f"{'Dice↑':<10} {'mIoU↑':<10} {'95HD↓':<10} {'ASD↓':<10}"

    print("\n" + "-" * 45)
    print(header)
    print(f"{metric[0]:<10.4f} {metric[1]:<10.4f} {metric[2]:<10.2f} {metric[3]:<10.2f}  (Mean)")
    print("-" * 45 + "\n")

    with open(os.path.join(test_save_path, '../performance.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{header}\n")
        f.write(f"{metric[0]:<10.4f} {metric[1]:<10.4f} {metric[2]:<10.2f} {metric[3]:<10.2f}  (Mean)\n")
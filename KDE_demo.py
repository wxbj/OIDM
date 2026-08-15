import os
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.net_factory import net_factory
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from configs import cfg

p_num = 500
bw_ad = 0.5
line_wid = 5


def get_ACDC_masks(output):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    return probs


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def plot_kde(BCP_feature, BCP_pred, labels, specific_c, f_dim, pic_num, save_base_dir):
    total_pixel = BCP_feature.shape[0]
    labeled_pixel = int(total_pixel / 2) + 1

    save_path = os.path.join(save_base_dir, "KDE_Plots", f"class_{specific_c}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    l_pred, u_pred = np.where(BCP_pred[:labeled_pixel, :] == specific_c), np.where(
        BCP_pred[labeled_pixel:, :] == specific_c)
    l_lab, u_lab = np.where(labels[:labeled_pixel, :] == specific_c), np.where(labels[labeled_pixel:, :] == specific_c)

    correct_cor_l = np.intersect1d(l_pred[0], l_lab[0])
    correct_cor_u = np.intersect1d(u_pred[0], u_lab[0]) + labeled_pixel

    pixel_num = min(len(correct_cor_l), len(correct_cor_u), p_num)
    print(f"Total {pixel_num} pixels for class {specific_c}")

    if pixel_num == 0:
        print(f"Not enough correct pixels found for class {specific_c}, skipping plot.")
        return

    BCP_feature_l = np.mean(BCP_feature[correct_cor_l[:pixel_num],], axis=1)
    BCP_feature_u = np.mean(BCP_feature[correct_cor_u[:pixel_num],], axis=1)

    method_name_list = [cfg.EXP]
    feature_list = [BCP_feature_l, BCP_feature_u]

    plt.figure()
    fig = plt.figure(figsize=(29, 4))
    sns.set_context("notebook", font_scale=2)

    for i in range(0, 1):
        plt.subplot(1, 1, i + 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
        sns.kdeplot(feature_list[0], bw_adjust=bw_ad, color='g', linewidth=line_wid, label="Labeled")
        sns.kdeplot(feature_list[1], bw_adjust=bw_ad, color='b', linewidth=line_wid, label="Unlabeled")
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.ylabel("Density")
        plt.title(method_name_list[i])
        plt.legend()

    plot_file = os.path.join(save_path, f"kde_test_mean{pic_num}_{cfg.DATASETS.LABEL_NUM}_{specific_c}.png")
    plt.savefig(plot_file)
    print(f"Save to: {plot_file}")
    plt.close('all')


def Inference():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)

    # 动态拼接模型路径
    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_ACDC")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"
    bcp_model_path = os.path.join(base_save_dir, exp_folder, cfg.TEST.STAGE_NAME, f"{cfg.MODEL.NAME}_best_model.pth")

    BCP_Net = net_factory(net_type=cfg.MODEL.NAME, in_chns=1, class_num=cfg.MODEL.NUM_CLASSES, mode="test")
    BCP_Net.load_state_dict(torch.load(bcp_model_path))
    print(f"init models' weight successfully from {bcp_model_path}")
    BCP_Net.eval()

    def worker_init_fn(worker_id):
        random.seed(cfg.SEED + worker_id)

    db_train = BaseDataSets(base_dir=cfg.DATASETS.ROOT_PATH,
                            split='train',
                            num=None,
                            transform=transforms.Compose([RandomGenerator(cfg.INPUT.PATCH_SIZE)]))

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(cfg.DATASETS.ROOT_PATH, cfg.DATASETS.LABEL_NUM)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))

    labeled_idx = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idxs, cfg.SOLVER.BATCH_SIZE,
                                          cfg.SOLVER.BATCH_SIZE - cfg.SOLVER.LABELED_BS)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    picture_number = 0
    for epoch_num in range(3):
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch = label_batch.detach().cpu().numpy()

            # Note: 这里的 BCP_Net 必须返回两个变量 (pred, BCP_feature)
            pred, BCP_feature = BCP_Net(volume_batch)

            B_pred = get_ACDC_masks(pred)

            f_dim, x_, y_ = BCP_feature.shape[1], BCP_feature.shape[2], BCP_feature.shape[3]

            BCP_feature = BCP_feature.permute(0, 2, 3, 1).contiguous()
            BCP_feature = BCP_feature.view(-1, f_dim)

            resized_label = np.zeros((cfg.SOLVER.BATCH_SIZE, x_, y_))
            for i in range(cfg.SOLVER.BATCH_SIZE):
                resized_label[i,] = cv2.resize(label_batch[i,].squeeze(), (x_, y_))

            label_batch = torch.from_numpy(resized_label).cuda()
            label = label_batch.view(-1, 1)
            BCP_pred = B_pred.view(-1, 1)

            BCP_feature = BCP_feature.detach().cpu().numpy()
            BCP_pred = BCP_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            # 绘制特定类别 (如 2) 的特征分布
            spi_c = 2
            save_base = os.path.join(base_save_dir, exp_folder)
            plot_kde(BCP_feature, BCP_pred, label, spi_c, f_dim, picture_number, save_base)
            picture_number += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BCP KDE Demo ACDC")
    parser.add_argument("-cfg", "--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.opts:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    Inference()

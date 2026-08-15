import argparse
import logging
import os
import random
import shutil
import sys
import setproctitle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import build_dataset, TwoStreamBatchSampler
from networks.net_factory import BCP_net
from utils import losses, val_2d
from configs import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dice_loss = losses.DiceLoss(n_classes=2)


def worker_init_fn(worker_id):
    random.seed(cfg.SEED + worker_id)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ISIC_masks(output):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    return probs


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_exact_gradient_conflict_mask(pre_l, lab_l, pre_u, plab_u, mask_ratio):
    batch_size, num_classes, img_x, img_y = pre_u.shape
    side_ratio = mask_ratio ** 0.5
    patch_x, patch_y = int(img_x * side_ratio), int(img_y * side_ratio)

    mask = torch.ones(batch_size, img_x, img_y).cuda()

    with torch.no_grad():
        prob_l = F.softmax(pre_l, dim=1)
        lab_l_onehot = F.one_hot(lab_l.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        grad_l = prob_l - lab_l_onehot
        g_l = grad_l.mean(dim=(2, 3))

        prob_u = F.softmax(pre_u, dim=1)
        plab_u_onehot = F.one_hot(plab_u.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        g_u = prob_u - plab_u_onehot

        conflict_map = -torch.sum(g_l.unsqueeze(2).unsqueeze(3) * g_u, dim=1)

        pool_map = F.avg_pool2d(conflict_map.unsqueeze(1), kernel_size=(patch_x, patch_y), stride=1).squeeze(1)

        for i in range(batch_size):
            max_idx = torch.argmax(pool_map[i].view(-1))
            w = max_idx // pool_map.shape[2]
            h = max_idx % pool_map.shape[2]

            mask[i, w:w + patch_x, h:h + patch_y] = 0

    return mask.long()


def generate_mask(img, seg_rate):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * seg_rate), int(img_y * seg_rate)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)

    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1))
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1))

    loss_ce = (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


def patients_to_slices(dataset, patiens_num):
    if "ISIC" in dataset.upper() or "isic" in dataset.lower():
        return int(patiens_num)
    else:
        print("Error")
        return int(patiens_num)


def get_train_val_splits(base_dir, val_ratio=0.2, seed=1337):
    train_list_path = os.path.join(base_dir, "train.list")
    with open(train_list_path, "r") as f:
        all_samples = [item.strip() for item in f.readlines() if item.strip()]

    random.Random(seed).shuffle(all_samples)

    val_size = int(len(all_samples) * val_ratio)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    return train_samples, val_samples


def pre_train(snapshot_path):
    base_lr = cfg.SOLVER.BASE_LR
    num_classes = cfg.MODEL.NUM_CLASSES
    max_iterations = cfg.SOLVER.PRE_ITERATIONS

    model = BCP_net(in_chns=3, class_num=num_classes).cuda()

    dataset_name = getattr(cfg.DATASETS, 'NAME', 'isic')

    train_samples, val_samples = get_train_val_splits(cfg.DATASETS.ROOT_PATH, val_ratio=0.2, seed=cfg.SEED)

    db_train = build_dataset(dataset_name=dataset_name,
                             base_dir=cfg.DATASETS.ROOT_PATH,
                             split="train",
                             patch_size=tuple(cfg.INPUT.PATCH_SIZE),
                             sample_list=train_samples)

    db_val = build_dataset(dataset_name=dataset_name,
                           base_dir=cfg.DATASETS.ROOT_PATH,
                           split="val",
                           patch_size=tuple(cfg.INPUT.PATCH_SIZE),
                           sample_list=val_samples)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(cfg.DATASETS.ROOT_PATH, cfg.DATASETS.LABEL_NUM)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.SOLVER.BATCH_SIZE,
                                          cfg.SOLVER.BATCH_SIZE - cfg.SOLVER.LABELED_BS)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = -1e6
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            labeled_img = volume_batch[:cfg.SOLVER.LABELED_BS]
            labeled_lab = label_batch[:cfg.SOLVER.LABELED_BS]

            half_bs = cfg.SOLVER.LABELED_BS // 2
            img_a, img_b = labeled_img[:half_bs], labeled_img[half_bs:]
            lab_a, lab_b = labeled_lab[:half_bs], labeled_lab[half_bs:]

            side_ratio = cfg.SOLVER.PRE_MASK_RATIO ** 0.5
            img_mask, loss_mask = generate_mask(img_a, seg_rate=side_ratio)

            net_input = img_a * img_mask + img_b * (1 - img_mask)

            outputs = model(net_input)
            out = outputs[0] if isinstance(outputs, tuple) else outputs

            loss_dice, loss_ce = mix_loss(out, lab_a, lab_b, loss_mask)
            loss = (loss_dice + loss_ce) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            lr_ = base_lr

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: lr: %f, loss: %f, mix_dice: %f, mix_ce: %f' % (
                iter_num, lr_, loss, loss_dice, loss_ce))

            if iter_num > 0 and iter_num % cfg.SOLVER.VAL_INTERVAL == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_image_isic(sampled_batch["image"], sampled_batch["label"], model,
                                                             classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_hd = np.mean(metric_list, axis=0)[1]

                composite_score = mean_dice - mean_hd

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                writer.add_scalar('info/val_composite_score', composite_score, iter_num)

                if composite_score > best_performance:
                    best_performance = composite_score
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : Score : %f, Dice: %f, HD: %f' % (
                    iter_num, composite_score, mean_dice, mean_hd))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(pre_snapshot_path, snapshot_path):
    base_lr = cfg.SOLVER.BASE_LR
    num_classes = cfg.MODEL.NUM_CLASSES
    max_iterations = cfg.SOLVER.MAX_ITERATIONS
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))

    model = BCP_net(in_chns=3, class_num=num_classes).cuda()
    ema_model = BCP_net(in_chns=3, class_num=num_classes, ema=True).cuda()

    dataset_name = getattr(cfg.DATASETS, 'NAME', 'isic')

    train_samples, val_samples = get_train_val_splits(cfg.DATASETS.ROOT_PATH, val_ratio=0.2, seed=cfg.SEED)

    db_train = build_dataset(dataset_name=dataset_name,
                             base_dir=cfg.DATASETS.ROOT_PATH,
                             split="train",
                             patch_size=tuple(cfg.INPUT.PATCH_SIZE),
                             sample_list=train_samples)

    db_val = build_dataset(dataset_name=dataset_name,
                           base_dir=cfg.DATASETS.ROOT_PATH,
                           split="val",
                           patch_size=tuple(cfg.INPUT.PATCH_SIZE),
                           sample_list=val_samples)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(cfg.DATASETS.ROOT_PATH, cfg.DATASETS.LABEL_NUM)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.SOLVER.BATCH_SIZE,
                                          cfg.SOLVER.BATCH_SIZE - cfg.SOLVER.LABELED_BS)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)

    writer = SummaryWriter(snapshot_path + '/log')
    model.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = -1e6
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img = volume_batch[:cfg.SOLVER.LABELED_BS]
            uimg = volume_batch[cfg.SOLVER.LABELED_BS:]
            lab = label_batch[:cfg.SOLVER.LABELED_BS]

            with torch.no_grad():
                pre_u_out = ema_model(uimg)
                pre_u = pre_u_out[0] if isinstance(pre_u_out, tuple) else pre_u_out
                plab_u = get_ISIC_masks(pre_u)

                pre_l_out = ema_model(img)
                pre_l = pre_l_out[0] if isinstance(pre_l_out, tuple) else pre_l_out

                mask = generate_exact_gradient_conflict_mask(pre_l, lab, pre_u, plab_u,
                                                             mask_ratio=cfg.SOLVER.SELF_MASK_RATIO)

            mask_unsqueeze = mask.unsqueeze(1)

            net_input_u_to_l = uimg * (1 - mask_unsqueeze) + img * mask_unsqueeze
            net_input_l_to_u = img * (1 - mask_unsqueeze) + uimg * mask_unsqueeze

            out_u_to_l_raw = model(net_input_u_to_l)
            out_l_to_u_raw = model(net_input_l_to_u)

            out_u_to_l = out_u_to_l_raw[0] if isinstance(out_u_to_l_raw, tuple) else out_u_to_l_raw
            out_l_to_u = out_l_to_u_raw[0] if isinstance(out_l_to_u_raw, tuple) else out_l_to_u_raw

            loss_dice_1, loss_ce_1 = mix_loss(out_u_to_l, lab, plab_u, mask)
            loss_dice_2, loss_ce_2 = mix_loss(out_l_to_u, plab_u, lab, mask)

            loss_ce = loss_ce_1 + loss_ce_2
            loss_dice = loss_dice_1 + loss_dice_2
            loss = loss_dice + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, cfg.SOLVER.EMA_ALPHA)

            lr_ = base_lr

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d: lr: %f, loss: %f, mix_dice: %f, mix_ce: %f' % (
                iter_num, lr_, loss, loss_dice, loss_ce))

            if iter_num > 0 and iter_num % cfg.SOLVER.VAL_INTERVAL == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_image_isic(sampled_batch["image"], sampled_batch["label"], model,
                                                             classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_hd = np.mean(metric_list, axis=0)[1]

                composite_score = mean_dice - mean_hd

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                writer.add_scalar('info/val_composite_score', composite_score, iter_num)

                if composite_score > best_performance:
                    best_performance = composite_score
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : Score : %f, Dice: %f, HD: %f' % (
                    iter_num, composite_score, mean_dice, mean_hd))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="BCP Training")
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

    if cfg.DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
    else:
        torch.backends.cudnn.benchmark = True

    try:
        setproctitle.setproctitle(f'{cfg.PROCTITLE}')
    except Exception:
        pass

    dataset_name = getattr(cfg.DATASETS, 'NAME', 'isic').upper()
    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{dataset_name}")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"

    pre_snapshot_path = os.path.join(base_save_dir, exp_folder, "pre_train")
    self_snapshot_path = os.path.join(base_save_dir, exp_folder, "self_train")

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy(__file__, self_snapshot_path)

    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    pre_train(pre_snapshot_path)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    self_train(pre_snapshot_path, self_snapshot_path)


if __name__ == "__main__":
    main()
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

from dataloaders.dataset import build_dataset, TwoStreamBatchSampler, get_labeled_indices
from networks.net_factory import BCP_net
from utils import losses
from utils.val_2d import calculate_metric_percase
from configs import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dice_loss = losses.DiceLoss(n_classes=2)


def worker_init_fn(worker_id):
    random.seed(cfg.SEED + worker_id)


def load_net(net, path):
    state = torch.load(str(path))
    if 'net' in state:
        net.load_state_dict(state['net'])
    else:
        net.load_state_dict(state)


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    if 'net' in state:
        net.load_state_dict(state['net'])
    else:
        net.load_state_dict(state)
    if 'opt' in state:
        optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_Promise12_masks(output):
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
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
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


def test_single_volume_promise(image, label_img, model, classes):
    image = image.squeeze(0).cpu().detach().numpy()  # [Z, 256, 256]
    label_img = label_img.squeeze(0).cpu().detach().numpy()  # [Z, 256, 256]
    prediction = np.zeros_like(label_img)

    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()

        model.eval()
        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            prediction[ind] = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label_img == i))
    return metric_list


def pre_train(snapshot_path):
    base_lr = cfg.SOLVER.BASE_LR
    num_classes = cfg.MODEL.NUM_CLASSES
    max_iterations = cfg.SOLVER.PRE_ITERATIONS

    model = BCP_net(in_chns=1, class_num=num_classes).cuda()

    db_train = build_dataset(dataset_name="promise12",
                             base_dir=cfg.DATASETS.ROOT_PATH,
                             split="train",
                             patch_size=cfg.INPUT.PATCH_SIZE)

    db_val = build_dataset(dataset_name="promise12",
                           base_dir=cfg.DATASETS.ROOT_PATH,
                           split="val",
                           patch_size=cfg.INPUT.PATCH_SIZE)

    labeled_idxs, unlabeled_idxs = get_labeled_indices(cfg.DATASETS.ROOT_PATH, cfg.DATASETS.LABEL_NUM)

    info_str = (f"Pre-training Dataset Loaded: "
                f"Total train slices: {len(db_train)} | "
                f"Labeled slices: {len(labeled_idxs)} | "
                f"Unlabeled slices: {len(unlabeled_idxs)} | "
                f"Val cases: {len(db_val)}")
    print(info_str)
    logging.info(info_str)

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
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

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
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/mix_dice', loss_dice.item(), iter_num)
            writer.add_scalar('info/mix_ce', loss_ce.item(), iter_num)

            logging.info('iteration %d: lr: %f, loss: %f, mix_dice: %f, mix_ce: %f' %
                         (iter_num, lr_, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num > 0 and iter_num % cfg.SOLVER.VAL_INTERVAL == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_promise(val_batch["image"], val_batch["label"], model,
                                                          classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_jc = np.mean(metric_list, axis=0)[1]
                mean_hd95 = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]

                composite_score = mean_dice + mean_jc - mean_hd95 - mean_asd

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i + 1), metric_list[class_i, 3], iter_num)

                writer.add_scalar('info/val_composite_score', composite_score, iter_num)

                if composite_score > best_performance:
                    best_performance = composite_score
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : Score : %f, Dice: %f, Jc: %f, HD95: %f, ASD: %f' % (
                    iter_num, composite_score, mean_dice, mean_jc, mean_hd95, mean_asd))
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

    model = BCP_net(in_chns=1, class_num=num_classes).cuda()
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True).cuda()

    db_train = build_dataset(dataset_name="promise12",
                             base_dir=cfg.DATASETS.ROOT_PATH,
                             split="train",
                             patch_size=cfg.INPUT.PATCH_SIZE)

    db_val = build_dataset(dataset_name="promise12",
                           base_dir=cfg.DATASETS.ROOT_PATH,
                           split="val",
                           patch_size=cfg.INPUT.PATCH_SIZE)

    labeled_idxs, unlabeled_idxs = get_labeled_indices(cfg.DATASETS.ROOT_PATH, cfg.DATASETS.LABEL_NUM)

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.SOLVER.BATCH_SIZE,
                                          cfg.SOLVER.BATCH_SIZE - cfg.SOLVER.LABELED_BS)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
    if os.path.exists(pre_trained_model):
        logging.info(f"Loading pretrained weights from: {pre_trained_model}")
        load_net(ema_model, pre_trained_model)
        load_net_opt(model, optimizer, pre_trained_model)
    else:
        raise FileNotFoundError(f"Cannot find pretrained model: {pre_trained_model}")

    writer = SummaryWriter(snapshot_path + '/log')
    model.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = -1e6
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            img = volume_batch[:cfg.SOLVER.LABELED_BS]
            uimg = volume_batch[cfg.SOLVER.LABELED_BS:]
            lab = label_batch[:cfg.SOLVER.LABELED_BS]

            with torch.no_grad():
                pre_u_out = ema_model(uimg)
                pre_u = pre_u_out[0] if isinstance(pre_u_out, tuple) else pre_u_out
                plab_u = get_Promise12_masks(pre_u)

                pre_l_out = ema_model(img)
                pre_l = pre_l_out[0] if isinstance(pre_l_out, tuple) else pre_l_out

                mask = generate_exact_gradient_conflict_mask(
                    pre_l, lab, pre_u, plab_u, mask_ratio=cfg.SOLVER.SELF_MASK_RATIO
                )
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
            writer.add_scalar('info/total_loss', loss.item(), iter_num)

            logging.info('iteration %d: lr: %f, loss: %f, mix_dice: %f, mix_ce: %f' %
                         (iter_num, lr_, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num > 0 and iter_num % cfg.SOLVER.VAL_INTERVAL == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_promise(val_batch["image"], val_batch["label"], model,
                                                          classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_jc = np.mean(metric_list, axis=0)[1]
                mean_hd95 = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]

                composite_score = mean_dice + mean_jc - mean_hd95 - mean_asd

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i + 1), metric_list[class_i, 3], iter_num)

                writer.add_scalar('info/val_composite_score', composite_score, iter_num)

                if composite_score > best_performance:
                    best_performance = composite_score
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : Score : %f, Dice: %f, Jc: %f, HD95: %f, ASD: %f' % (
                    iter_num, composite_score, mean_dice, mean_jc, mean_hd95, mean_asd))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="BCP Training Promise12")
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
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
    else:
        torch.backends.cudnn.benchmark = True

    try:
        setproctitle.setproctitle(f'{cfg.PROCTITLE}')
    except Exception:
        pass

    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_Promise12")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"

    pre_snapshot_path = os.path.join(base_save_dir, exp_folder, "pre_train")
    self_snapshot_path = os.path.join(base_save_dir, exp_folder, "self_train")

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy(__file__, self_snapshot_path)

    print("Starting Pre-training...")
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    pre_train(pre_snapshot_path)

    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    print("Starting Self-training...")
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    self_train(pre_snapshot_path, self_snapshot_path)


if __name__ == "__main__":
    main()
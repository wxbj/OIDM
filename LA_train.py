import sys
import os
import random
import shutil
import argparse
import logging
import setproctitle

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import losses, test_3d_patch
from dataloaders.dataset import build_dataset, TwoStreamBatchSampler
from networks.net_factory import net_factory
from configs import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def worker_init_fn(worker_id):
    random.seed(cfg.SEED + worker_id)


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    if 'net' in state:
        net.load_state_dict(state['net'])
    else:
        net.load_state_dict(state)
    if 'opt' in state:
        optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    if 'net' in state:
        net.load_state_dict(state['net'])
    else:
        net.load_state_dict(state)


def get_LA_masks(output):
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


def generate_exact_gradient_conflict_mask_3d(pre_l, lab_l, pre_u, plab_u, mask_ratio):
    batch_size, num_classes, dim1, dim2, dim3 = pre_u.shape
    side_ratio = mask_ratio ** (1.0 / 3.0)
    patch_1 = int(dim1 * side_ratio)
    patch_2 = int(dim2 * side_ratio)
    patch_3 = int(dim3 * side_ratio)

    mask = torch.ones(batch_size, dim1, dim2, dim3, device=pre_u.device)

    with torch.no_grad():
        prob_l = F.softmax(pre_l, dim=1)
        lab_l_onehot = F.one_hot(lab_l.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        grad_l = prob_l - lab_l_onehot

        g_l = grad_l.mean(dim=(2, 3, 4))

        prob_u = F.softmax(pre_u, dim=1)
        plab_u_onehot = F.one_hot(plab_u.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        g_u = prob_u - plab_u_onehot

        conflict_map = -torch.sum(g_l.unsqueeze(2).unsqueeze(3).unsqueeze(4) * g_u, dim=1)

        pool_map = F.avg_pool3d(conflict_map.unsqueeze(1), kernel_size=(patch_1, patch_2, patch_3), stride=1).squeeze(1)

        for i in range(batch_size):
            max_idx = torch.argmax(pool_map[i].view(-1))
            S_Y = pool_map.shape[2]
            S_Z = pool_map.shape[3]

            d1_idx = max_idx // (S_Y * S_Z)
            rem = max_idx % (S_Y * S_Z)
            d2_idx = rem // S_Z
            d3_idx = rem % S_Z

            mask[i, d1_idx:d1_idx + patch_1, d2_idx:d2_idx + patch_2, d3_idx:d3_idx + patch_3] = 0

    return mask.long()


def generate_mask_3d(img, seg_rate):
    batch_size, channel, dim1, dim2, dim3 = img.shape
    loss_mask = torch.ones(batch_size, dim1, dim2, dim3).cuda()
    mask = torch.ones(dim1, dim2, dim3).cuda()

    patch_1 = int(dim1 * seg_rate)
    patch_2 = int(dim2 * seg_rate)
    patch_3 = int(dim3 * seg_rate)

    w = np.random.randint(0, dim1 - patch_1)
    h = np.random.randint(0, dim2 - patch_2)
    d = np.random.randint(0, dim3 - patch_3)

    mask[w:w + patch_1, h:h + patch_2, d:d + patch_3] = 0
    loss_mask[:, w:w + patch_1, h:h + patch_2, d:d + patch_3] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)

    patch_mask = 1 - mask
    num_classes = output.shape[1]

    img_l_onehot = F.one_hot(img_l, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    patch_l_onehot = F.one_hot(patch_l, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    mask_uns = mask.unsqueeze(1)  # [B, 1, H, W, D]
    patch_mask_uns = patch_mask.unsqueeze(1)  # [B, 1, H, W, D]

    intersect_1 = torch.sum(output_soft * img_l_onehot * mask_uns, dim=(2, 3, 4))
    denominator_1 = torch.sum((output_soft + img_l_onehot) * mask_uns, dim=(2, 3, 4))
    dice_1 = 1.0 - (2.0 * intersect_1 + 1e-16) / (denominator_1 + 1e-16)
    dice_1 = dice_1[:, 1:].mean()

    intersect_2 = torch.sum(output_soft * patch_l_onehot * patch_mask_uns, dim=(2, 3, 4))
    denominator_2 = torch.sum((output_soft + patch_l_onehot) * patch_mask_uns, dim=(2, 3, 4))
    dice_2 = 1.0 - (2.0 * intersect_2 + 1e-16) / (denominator_2 + 1e-16)
    dice_2 = dice_2[:, 1:].mean()

    loss_dice = dice_1 + dice_2

    loss_ce_1 = (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce_2 = (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss_ce = loss_ce_1 + loss_ce_2

    return loss_dice, loss_ce


def pre_train(snapshot_path):
    model = net_factory(net_type=cfg.MODEL.NAME, in_chns=1, class_num=cfg.MODEL.NUM_CLASSES, mode="train").cuda()

    db_train = build_dataset(dataset_name="la",
                             base_dir=cfg.DATASETS.ROOT_PATH,
                             split="train",
                             patch_size=tuple(cfg.INPUT.PATCH_SIZE))

    total_samples = len(db_train)
    labeled_idxs = list(range(cfg.DATASETS.LABEL_NUM))
    unlabeled_idxs = list(range(cfg.DATASETS.LABEL_NUM, total_samples))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.SOLVER.BATCH_SIZE,
                                          cfg.SOLVER.BATCH_SIZE - cfg.SOLVER.LABELED_BS)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=0.9, weight_decay=0.0001)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_performance = -1e6
    max_epoch = cfg.SOLVER.PRE_ITERATIONS // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            labeled_img = volume_batch[:cfg.SOLVER.LABELED_BS]
            labeled_lab = label_batch[:cfg.SOLVER.LABELED_BS]

            half_bs = cfg.SOLVER.LABELED_BS // 2
            img_a, img_b = labeled_img[:half_bs], labeled_img[half_bs:]
            lab_a, lab_b = labeled_lab[:half_bs], labeled_lab[half_bs:]

            side_ratio = cfg.SOLVER.PRE_MASK_RATIO ** (1.0 / 3.0)
            img_mask, loss_mask = generate_mask_3d(img_a, seg_rate=side_ratio)

            net_input = img_a * img_mask + img_b * (1 - img_mask)

            outputs = model(net_input)
            out = outputs[0] if isinstance(outputs, tuple) else outputs

            loss_dice, loss_ce = mix_loss(out, lab_a, lab_b, loss_mask)
            loss = (loss_dice + loss_ce) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            lr_ = cfg.SOLVER.BASE_LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/mix_dice', loss_dice.item(), iter_num)
            writer.add_scalar('info/mix_ce', loss_ce.item(), iter_num)

            logging.info('iteration %d: lr: %f, loss: %f, mix_dice: %f, mix_ce: %f' % (
                iter_num, lr_, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num > 0 and iter_num % cfg.SOLVER.VAL_INTERVAL == 0:
                model.eval()
                avg_metric = test_3d_patch.var_all_case_LA(model, num_classes=cfg.MODEL.NUM_CLASSES,
                                                           patch_size=tuple(cfg.INPUT.PATCH_SIZE), stride_xy=18,
                                                           stride_z=4)
                mean_dice, mean_jc, mean_hd95, mean_asd = avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3]
                composite_score = mean_dice + mean_jc - mean_hd95 - mean_asd

                if composite_score > best_performance:
                    best_performance = composite_score
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_best_path))

                logging.info('iteration %d : Score: %f, Dice: %f, Jc: %f, HD95: %f, ASD: %f' % (
                    iter_num, composite_score, mean_dice, mean_jc, mean_hd95, mean_asd))

                writer.add_scalar('4_Var_dice/Dice', mean_dice, iter_num)
                writer.add_scalar('4_Var_dice/JC', mean_jc, iter_num)
                writer.add_scalar('4_Var_dice/HD95', mean_hd95, iter_num)
                writer.add_scalar('4_Var_dice/ASD', mean_asd, iter_num)
                writer.add_scalar('4_Var_dice/Composite_Score', composite_score, iter_num)
                model.train()

            if iter_num >= cfg.SOLVER.PRE_ITERATIONS:
                break
        if iter_num >= cfg.SOLVER.PRE_ITERATIONS:
            iterator.close()
            break
    writer.close()


def self_train(pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=cfg.MODEL.NAME, in_chns=1, class_num=cfg.MODEL.NUM_CLASSES, mode="train").cuda()
    ema_model = net_factory(net_type=cfg.MODEL.NAME, in_chns=1, class_num=cfg.MODEL.NUM_CLASSES, mode="train").cuda()
    for param in ema_model.parameters():
        param.detach_()

    db_train = build_dataset(dataset_name="la",
                             base_dir=cfg.DATASETS.ROOT_PATH,
                             split="train",
                             patch_size=tuple(cfg.INPUT.PATCH_SIZE))

    total_samples = len(db_train)
    labeled_idxs = list(range(cfg.DATASETS.LABEL_NUM))
    unlabeled_idxs = list(range(cfg.DATASETS.LABEL_NUM, total_samples))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.SOLVER.BATCH_SIZE,
                                          cfg.SOLVER.BATCH_SIZE - cfg.SOLVER.LABELED_BS)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{cfg.MODEL.NAME}_best_model.pth')
    if os.path.exists(pretrained_model):
        load_net(model, pretrained_model)
        load_net(ema_model, pretrained_model)
    else:
        raise FileNotFoundError(f"Cannot find pretrained model: {pretrained_model}")

    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_performance = -1e6
    max_epoch = cfg.SOLVER.MAX_ITERATIONS // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            img = volume_batch[:cfg.SOLVER.LABELED_BS]
            uimg = volume_batch[cfg.SOLVER.LABELED_BS:]
            lab = label_batch[:cfg.SOLVER.LABELED_BS]

            with torch.no_grad():
                pre_u_out = ema_model(uimg)
                pre_u = pre_u_out[0] if isinstance(pre_u_out, tuple) else pre_u_out
                plab_u = get_LA_masks(pre_u)

                pre_l_out = ema_model(img)
                pre_l = pre_l_out[0] if isinstance(pre_l_out, tuple) else pre_l_out

                mask = generate_exact_gradient_conflict_mask_3d(pre_l, lab, pre_u, plab_u,
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

            lr_ = cfg.SOLVER.BASE_LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/mix_dice', loss_dice.item(), iter_num)
            writer.add_scalar('info/mix_ce', loss_ce.item(), iter_num)

            logging.info('iteration %d: lr: %f, loss: %f, mix_dice: %f, mix_ce: %f' % (
                iter_num, lr_, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num > 0 and iter_num % cfg.SOLVER.VAL_INTERVAL == 0:
                model.eval()
                avg_metric = test_3d_patch.var_all_case_LA(model, num_classes=cfg.MODEL.NUM_CLASSES,
                                                           patch_size=tuple(cfg.INPUT.PATCH_SIZE), stride_xy=18,
                                                           stride_z=4)

                mean_dice, mean_jc, mean_hd95, mean_asd = avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3]
                composite_score = mean_dice + mean_jc - mean_hd95 - mean_asd

                if composite_score > best_performance:
                    best_performance = composite_score
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(cfg.MODEL.NAME))
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_best_path))

                logging.info('iteration %d : Score: %f, Dice: %f, Jc: %f, HD95: %f, ASD: %f' % (
                    iter_num, composite_score, mean_dice, mean_jc, mean_hd95, mean_asd))

                writer.add_scalar('4_Var_dice/Dice', mean_dice, iter_num)
                writer.add_scalar('4_Var_dice/JC', mean_jc, iter_num)
                writer.add_scalar('4_Var_dice/HD95', mean_hd95, iter_num)
                writer.add_scalar('4_Var_dice/ASD', mean_asd, iter_num)
                writer.add_scalar('4_Var_dice/Composite_Score', composite_score, iter_num)
                model.train()

            if iter_num % cfg.SOLVER.VAL_INTERVAL == 1:
                ins_width = 2
                B, C, H, W, D = out_u_to_l.size()
                snapshot_img = torch.zeros(size=(D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype=torch.float32)

                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2 * H + ins_width:2 * H + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * H + 2 * ins_width:3 * H + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1

                mixl_lab = lab * mask + plab_u * (1 - mask)
                mixu_lab = plab_u * mask + lab * (1 - mask)

                outputs_l_soft = F.softmax(out_u_to_l, dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1).cpu().detach()
                target = mixl_lab[0, ...].permute(2, 0, 1).cpu().detach()
                train_img = net_input_u_to_l[0, 0, ...].permute(2, 0, 1).cpu().detach()

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = snapshot_img[:, 0, :H, :W]
                snapshot_img[:, 2, :H, :W] = snapshot_img[:, 0, :H, :W]

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(out_l_to_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1).cpu().detach()
                target = mixu_lab[0, ...].permute(2, 0, 1).cpu().detach()
                train_img = net_input_l_to_u[0, 0, ...].permute(2, 0, 1).cpu().detach()

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = snapshot_img[:, 0, :H, :W]
                snapshot_img[:, 2, :H, :W] = snapshot_img[:, 0, :H, :W]

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= cfg.SOLVER.MAX_ITERATIONS:
                break
        if iter_num >= cfg.SOLVER.MAX_ITERATIONS:
            iterator.close()
            break
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="BCP Training LA")
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

    base_save_dir = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_LA")
    exp_folder = f"{cfg.EXP}_{cfg.DATASETS.LABEL_NUM}_labeled"

    pre_snapshot_path = os.path.join(base_save_dir, exp_folder, "pre_train")
    self_snapshot_path = os.path.join(base_save_dir, exp_folder, "self_train")

    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy(__file__, self_snapshot_path)

    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    pre_train(pre_snapshot_path)

    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    self_train(pre_snapshot_path, self_snapshot_path)


if __name__ == "__main__":
    main()
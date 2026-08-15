import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt) * 100.0
        jc = metric.binary.jc(pred, gt) * 100.0
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0.0, 0.0, 128.0, 128.0


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output) > 1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def calculate_metric_percase_isic(pred, gt):
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


def test_single_image_isic(image, label, model, classes):
    image = image.cuda()
    if image.ndim == 4 and image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)

    label = label.squeeze(0).cpu().detach().numpy()

    model.eval()
    with torch.no_grad():
        output = model(image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_isic(prediction == i, label == i))
    return metric_list
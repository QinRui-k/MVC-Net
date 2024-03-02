import math
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label



def getLargestCC(segmentation):
    labels = label(segmentation)
    # assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC



def test_all_case_fuse(net1, net2, net3, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                       save_result=True, test_save_path=None, preproc_fn=None):
    total_metric_dice = 0.0
    for image_path in (image_list):
        image_paths = '.data/CMT-TS' + "/" + image_path + '/' + image_path + '.nii'
        # print(image_paths)
        label_paths = '.data/CMT-TS' + "/" + image_path + '/Label.nii'
        # print(label_paths)
        image1 = nib.load(image_paths)
        image = np.array(image1.dataobj)
        # print(image.shape)
        label1 = nib.load(label_paths)
        label = np.array(label1.dataobj)
        image2 = np.where(image > 1000)
        minx, maxx = np.min(image2[0]), np.max(image2[0])
        miny, maxy = np.min(image2[1]), np.max(image2[1])
        minz, maxz = np.min(image2[2]), np.max(image2[2])
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]

        lower_percentile = np.percentile(image, 5)
        upper_percentile = np.percentile(image, 95)
        image = np.clip(image, lower_percentile, upper_percentile)

        image = (image - lower_percentile) / (upper_percentile - lower_percentile)

        if preproc_fn is not None:
            image = preproc_fn(image)
        image1 = image
        image2 = np.transpose(image, (1, 2, 0))
        image3 = np.transpose(image, (2, 0, 1))

        prediction1, score_map1 = test_single_case(net1, image1, stride_xy, stride_z, patch_size,
                                                   num_classes=num_classes)
        prediction2, score_map2 = test_single_case(net2, image2, stride_xy, stride_z, patch_size,
                                                   num_classes=num_classes)
        prediction2 = np.transpose(prediction2, (2, 0, 1))
        score_map2 = np.transpose(score_map2, (0, 3, 1, 2))
        prediction3, score_map3 = test_single_case(net3, image3, stride_xy, stride_z, patch_size,
                                                   num_classes=num_classes)
        prediction3 = np.transpose(prediction3, (1, 2, 0))
        score_map3 = np.transpose(score_map3, (0, 2, 3, 1))

        score_map = score_map1+score_map2+score_map3
        prediction = np.argmax(score_map, axis=0)

        prediction_mask = (prediction > 0).astype(int)
        prediction_mask = getLargestCC(prediction_mask)

        prediction1 = (prediction == 1).astype(int)
        prediction2 = (prediction == 2).astype(int)
        prediction3 = (prediction == 3).astype(int)
        prediction4 = (prediction == 4).astype(int)

        prediction1 = prediction1 * prediction_mask
        prediction2 = prediction2 * prediction_mask
        prediction3 = prediction3 * prediction_mask
        prediction4 = prediction4 * prediction_mask

        prediction1 = getLargestCC(prediction1)
        prediction2 = getLargestCC(prediction2)
        prediction3 = getLargestCC(prediction3)
        prediction4 = getLargestCC(prediction4)
        prediction = prediction1 * 1 + prediction2 * 2 + prediction3 * 3 + prediction4 * 4

        if np.sum(prediction) == 0:
            single_metric_dice = (0, 0, 0, 0)
        else:
            single_metric_dice = cal_dice(prediction, label[:], num=num_classes)
        single_metric_dice = np.clip(single_metric_dice, 0, 1)
        total_metric_dice += np.asarray(single_metric_dice)
        print(np.mean(single_metric_dice))
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + image_path + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)),
                     test_save_path + image_path + "_score.nii.gz")

    avg_metric_dice = total_metric_dice / len(image_list)

    return avg_metric_dice


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map



def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice



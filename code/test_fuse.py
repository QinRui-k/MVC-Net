import os
import torch
from networks.vnet import VNet
from test_util import test_all_case_fuse



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

num_classes = 5

with open('.data/CMT-TS/test.txt', 'r') as f:
    image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]


def test_calculate_metric_fuse(model_name):
    snapshot_path = "../model/"+model_name+"/"
    test_save_path = "../model/prediction/" + model_name + "_post/"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net1 = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'model1_best.pth')
    net1.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net1.eval()

    net2 = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'model2_best.pth')
    net2.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net2.eval()

    net3 = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'model3_best.pth')
    net3.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net3.eval()

    avg_metric_dice1 = test_all_case_fuse(net1, net2, net3, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 112), stride_xy=16, stride_z=16,
                               save_result=True, test_save_path=test_save_path)
    return avg_metric_dice1



if __name__ == '__main__':
    model_name = 'MVC-Net'

    avg_metric_dice1 = test_calculate_metric_fuse(model_name)
    print(avg_metric_dice1)



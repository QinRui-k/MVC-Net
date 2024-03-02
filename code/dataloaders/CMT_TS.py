import random
import torch
import numpy as np
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib




class CMT_TS(Dataset):

    def __init__(self, transform=None):
        self.labeled_base_dir = './data/CMT-TS'
        self.unlabeled_base_dir = './data/CMT-TS'
        self.transform = transform
        self.labeled_image_list = []

        with open(self.labeled_base_dir + '/train.txt', 'r') as f:
            self.labeled_image_list = f.readlines()
            self.labeled_image_list = [item.replace('\n', '') for item in self.labeled_image_list]

        with open(self.unlabeled_base_dir + '/train.txt', 'r') as f:
            self.unlabeled_image_list = f.readlines()
            self.unlabeled_image_list = [item.replace('\n', '') for item in self.unlabeled_image_list]

        print("total {} labled samples".format(len(self.labeled_image_list)))
        print("total {} unlabled samples".format(len(self.unlabeled_image_list)))

    def __len__(self):
        return len(self.labeled_image_list)

    def __getitem__(self, idx):
        labeled_image_name = self.labeled_image_list[idx]
        unlabeled_image_name = self.unlabeled_image_list[random.randint(0,len(self.unlabeled_image_list)-1)]
        print("--------------------------------------------------------------------------------------------")
        print("labeled_image name: ", labeled_image_name)
        print("unlabeled_image name: ", unlabeled_image_name)


        image_path = self.labeled_base_dir+"/"+labeled_image_name+'/'+labeled_image_name+'.nii'
        label_path = self.labeled_base_dir+"/"+labeled_image_name+'/Label.nii'
        image1 = nib.load(image_path)
        image = np.array(image1.dataobj)
        label1 = nib.load(label_path)
        label = np.array(label1.dataobj)

        image2 = np.where(image > 1000)
        minx, maxx = np.min(image2[0]), np.max(image2[0])
        miny, maxy = np.min(image2[1]), np.max(image2[1])
        minz, maxz = np.min(image2[2]), np.max(image2[2])
        image = image[minx:maxx,miny:maxy,minz:maxz]
        label = label[minx:maxx,miny:maxy,minz:maxz]


        unlabeled_image_path = self.unlabeled_base_dir+"/"+unlabeled_image_name+'/'+unlabeled_image_name+'.nii'
        unimage1 = nib.load(unlabeled_image_path)
        unlabeled_image = np.array(unimage1.dataobj)
        uimage = np.where(unlabeled_image > 500)
        minx, maxx = np.min(uimage[0]), np.max(uimage[0])
        miny, maxy = np.min(uimage[1]), np.max(uimage[1])
        minz, maxz = np.min(uimage[2]), np.max(uimage[2])
        unlabeled_image = unlabeled_image[minx:maxx, miny:maxy, minz:maxz]



        sample = {'image': image, 'label': label, 'unlabel_image': unlabeled_image}

        if self.transform:
            sample = self.transform(sample)

        return sample




class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, unlabel_image = sample['image'], sample['label'],sample['unlabel_image']

        lower_percentile = np.percentile(image, 5)
        upper_percentile = np.percentile(image, 95)
        image = np.clip(image, lower_percentile, upper_percentile)

        image = (image-lower_percentile)/(upper_percentile-lower_percentile)

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]




        lower_percentile = np.percentile(unlabel_image, 5)
        upper_percentile = np.percentile(unlabel_image, 95)
        unlabel_image = np.clip(unlabel_image, lower_percentile, upper_percentile)

        unlabel_image = (unlabel_image - lower_percentile) / (upper_percentile - lower_percentile)

        # pad the sample if necessary
        if unlabel_image.shape[0] <= self.output_size[0] or unlabel_image.shape[1] <= self.output_size[1] or unlabel_image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - unlabel_image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - unlabel_image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - unlabel_image.shape[2]) // 2 + 3, 0)
            unlabel_image = np.pad(unlabel_image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


        (w, h, d) = unlabel_image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        unlabel_image = unlabel_image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]



        return {'image': image, 'label': label, 'unlabel_image': unlabel_image}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label, unlabel_image = sample['image'], sample['label'],sample['unlabel_image']
        k = np.random.randint(0, 4)
        unlabel_image = np.rot90(unlabel_image, k)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label, 'unlabel_image': unlabel_image}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.02):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label, unlabel_image = sample['image'], sample['label'], sample['unlabel_image']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        noise_un = np.clip(self.sigma * np.random.randn(unlabel_image.shape[0], unlabel_image.shape[1], unlabel_image.shape[2]), -2*self.sigma, 2*self.sigma)
        unlabel_image = unlabel_image + noise_un
        return {'image': image, 'label': label, 'unlabel_image': unlabel_image}





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, unlabel_image = sample['image'], sample['label'],sample['unlabel_image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        unlabel_image = unlabel_image.reshape(1, unlabel_image.shape[0], unlabel_image.shape[1], unlabel_image.shape[2]).astype(np.float32)

        return {'image': image, 'label': label, 'unlabel_image': unlabel_image}



















class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

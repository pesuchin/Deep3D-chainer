# Dataset
import os
import numpy as np
from PIL import Image
import six
from pathlib import Path

import chainer
from chainer.dataset import dataset_mixin
from chainer import training
from chainer import iterators
from chainer.training import extensions
from chainer.backends import cuda

from deep3d import Deep3D


def resize_image(img, size, vgg_prepare_flag=True):
    w, h = img.size
    img = img.resize((int(w * (size[1] / h)), size[1]))
    img = np.array(img, dtype=np.float32)
    if vgg_prepare_flag:
        img = img[:, :, ::-1]
        img -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img = img.transpose(2, 0, 1)
    ch, h, w = img.shape
    offset_w = (w - size[0]) // 2
    img = img[:,:,offset_w:offset_w+size[0]]
    return img


def _read_kitti_as_array(path_left, path_right, size=(768, 256)):
    # kitti size: (1226, 370)
    left = Image.open(path_left)
    right = Image.open(path_right)
    if np.random.random() > 0.5:
        left, right, original_left = augument_random_crop(left, right, crop_size=size)
    else:
        original_left = resize_image(left, size, vgg_prepare_flag=False)
        left = resize_image(left, size)
        right = resize_image(right, size)
    return left, right, original_left


def augument_random_crop(left, right, crop_size=(768, 256)):
    left = np.array(left, dtype=np.float32)
    original_left = left.transpose(2, 0, 1)
    left = left[:, :, ::-1]
    left -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    left = left.transpose(2, 0, 1)
    right = np.array(right, dtype=np.float32)
    right = right.transpose(2, 0, 1)
    _, h, w = left.shape

    # randomly select crop start possition (top and left)
    top = np.random.randint(0, h - crop_size[1])
    left_position = np.random.randint(0, w - crop_size[0])

    # calc top and left bound
    bottom = top + crop_size[1]
    right_position = left_position + crop_size[0]

    # crop image
    left = left[:, top:bottom, left_position:right_position]
    right = right[:, top:bottom, left_position:right_position]
    original_left = original_left[:, top:bottom, left_position:right_position]
    return left, right, original_left


def augment_image_pair(left_image, right_image):
    # randomly shift gamma
    random_gamma = np.random.uniform(0.8, 1.2)
    left_image_aug  = left_image  ** random_gamma
    right_image_aug = right_image ** random_gamma

    # randomly shift brightness
    random_brightness = np.random.uniform(0.5, 2.0)
    left_image_aug  =  left_image_aug * random_brightness
    right_image_aug = right_image_aug * random_brightness

    # randomly shift color
    random_colors = np.random.uniform(0.8, 1.2, 3)
    left_image_aug  *= random_colors[:, None, None]
    right_image_aug *= random_colors[:, None, None]

    # saturate
    left_image_aug  = np.clip(left_image_aug,  0., 1.)
    right_image_aug = np.clip(right_image_aug, 0., 1.)

    return left_image_aug, right_image_aug


def prepare_dataset_path(folder_path):
    current_dir = Path.cwd()
    left_datasets_dir = current_dir.parent / folder_path / 'left'
    right_datasets_dir = current_dir.parent / folder_path / 'right'
    left_file_names = left_datasets_dir.glob('**/*.png')
    right_file_names = right_datasets_dir.glob('**/*.png')

    path_left_imgs = []
    path_right_imgs = []
    for l_img, r_img in zip(left_file_names, right_file_names):
        path_left_imgs.append(str(l_img))
        path_right_imgs.append(str(r_img))

    paths = list(zip(path_left_imgs, path_right_imgs))
    
    return paths


def horizontal_flip(left, right, original_left):
    left = left[:, :, ::-1]
    right = right[:, :, ::-1]
    original_left = original_left[:, :, ::-1]
    return left, right, original_left


class KittiDataset(dataset_mixin.DatasetMixin):

    def __init__(self, paths, root='.', augmentation=True):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip().split(' ') for path in paths_file]
        self._paths = paths
        self._root = root
        self.augmentation = augmentation

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path_left, path_right = self._paths[i]
        path_left  = os.path.join(self._root, path_left).replace('.jpg', '.png')
        path_right = os.path.join(self._root, path_right).replace('.jpg', '.png')
        left, right, original_left = _read_kitti_as_array(path_left, path_right, size=(352, 256))
        if self.augmentation:
            #if np.random.random() > 0.5:
            #    left, right = augment_image_pair(left, right)
            if np.random.random() > 0.5:
                left, right, original_left = horizontal_flip(left, right, original_left)
        left /= 255.
        right /= 255.
        original_left /= 255.
        return left, right, original_left


def main():
    iteration = 200000
    out_dir = './result/'
    batchsize = 16
    gpu = 0
    size = (352, 256)

    snapshot_interval = (2000, 'iteration')
    report_interval = (10, 'iteration')
    rate_change_trigger = (20000, 'iteration')

    model = Deep3D(batchsize, size)
    optimizer = chainer.optimizers.Adam(alpha=0.00002)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer_hooks.GradientNoise(0.01))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

    train_paths = prepare_dataset_path('datasets')
    train_dataset = KittiDataset(train_paths, '')

    train_iter = iterators.MultiprocessIterator(train_dataset, batchsize, repeat=True, shuffle=True, n_processes=4)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out_dir)
    trainer.extend(extensions.ExponentialShift('alpha', rate=0.1), trigger=rate_change_trigger)
    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=report_interval))
    trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'validation/main/loss']),
                trigger=report_interval)

    trainer.run()


if __name__ == '__main__':
    main()

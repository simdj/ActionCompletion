import random
import PIL
import math
import numbers
import cv2

import numpy as np


import torch
from torchvision import datasets, models, transforms



from datasets.video_to_frames import UCF101Dataset
from datasets.rgbd_ac import RGBD_AC_Dataset
from datasets.self_supervised_data import Self_Supervised_Dataset, Union_Dataset
from config import CONFIG


# reference: https://github.com/hassony2/torch_videovision

def crop_clip(clip, min_h, min_w, h, w):
    # reference: https://github.com/hassony2/torch_videovision
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped

def resize_clip(clip, size, interpolation='bilinear'):
    # reference: https://github.com/hassony2/torch_videovision
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


class VideoRandomResizedCrop(object):
    """
    # reference: https://github.com/hassony2/torch_videovision

    Crop the given list of PIL Images to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(clip, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (list of PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        if isinstance(clip[0], np.ndarray):
            height, width, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            width, height = clip[0].size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, clip):
        """
        Args:
            clip: list of img (PIL Image): Image to be cropped and resized.
        Returns:
            list of PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        imgs=crop_clip(clip,i,j,h,w)
        return resize_clip(clip,self.size,self.interpolation)
        # return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


def get_dataset(data_dir, task_spec=None):
    input_size = CONFIG.IMAGE_SIZE
    split = CONFIG.DATA.SPLIT
    
    video_transforms = transforms.Compose([
        VideoRandomResizedCrop(input_size)
    ])

    data_transforms = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
        "val": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }   

    
    if CONFIG.DATA.DATASET=='ucf101':
        class_idx_filename = 'completion_all_classInd.txt'
        # class_idx_filename = 'completion_blowing_classInd.txt'        
        train_dataset = UCF101Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=True, transforms_=data_transforms["train"], video_transforms=_video_transforms)
        test_dataset  = UCF101Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=False, transforms_=data_transforms["val"])
    elif CONFIG.DATA.DATASET=='rgbd_ac':
        class_idx_filename = 'completion_all_classInd.txt'
        # class_idx_filename = 'completion_open_classInd.txt'
        # data_dir : ../data/RGBD-AC
        train_dataset = RGBD_AC_Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN,  train=True, transforms_=data_transforms["train"], video_transforms_=video_transforms)
        test_dataset  = RGBD_AC_Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN,  train=False, transforms_=data_transforms["val"])
    # elif CONFIG.DATA.DATASET =='self':
    #     train_dataset = Self_Supervised_Dataset(data_dir, video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=True, class_num=CONFIG.DATA.ACTION_CLASS_NUM, transforms_=data_transforms['train'])
    #     test_dataset  = Self_Supervised_Dataset(data_dir, video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=False, class_num=CONFIG.DATA.ACTION_CLASS_NUM, transforms_=data_transforms['val'])
    else:
        raise ValueError('Specify data(ucf101/hmdb51/rgbd-ac/self')


    if CONFIG.SELF_LEARN.USE:
        self_learn_class_idx = max(train_dataset.class_set)+1
        ss_dataset = Self_Supervised_Dataset(data_dir, video_len=CONFIG.SELF_LEARN.VIDEO_LEN, class_idx=self_learn_class_idx, train=True, class_num=CONFIG.DATA.ACTION_CLASS_NUM, transforms_=data_transforms['train'])
        train_dataset = Union_Dataset(train_dataset, ss_dataset)


    return train_dataset, test_dataset



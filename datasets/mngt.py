

import torch
from torchvision import datasets, models, transforms



from datasets.video_to_frames import UCF101Dataset
from datasets.rgbd_ac import RGBD_AC_Dataset
from datasets.self_supervised_data import Self_Supervised_Dataset, Union_Dataset
from config import CONFIG

def get_dataset(data_dir, task_spec=None):
    input_size = CONFIG.IMAGE_SIZE
    split = CONFIG.DATA.SPLIT
    
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
        train_dataset = UCF101Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=True, transforms_=data_transforms["train"])
        test_dataset  = UCF101Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=False, transforms_=data_transforms["val"])
    elif CONFIG.DATA.DATASET=='rgbd_ac':
        class_idx_filename = 'completion_all_classInd.txt'
        # class_idx_filename = 'completion_open_classInd.txt'
        # data_dir : ../data/RGBD-AC
        train_dataset = RGBD_AC_Dataset(data_dir, split=split, class_idx_filename=class_idx_filename, class_num=CONFIG.DATA.ACTION_CLASS_NUM, max_video_len=CONFIG.SELF_LEARN.VIDEO_LEN,  train=True, transforms_=data_transforms["train"])
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



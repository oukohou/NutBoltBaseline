# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-11-20'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
construct a dataloader for datasets.
"""

import os

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class ReadImageDataset(Dataset):
    def __init__(self, image_list_, base_path,  mode='train', input_size=64,
                 use_base_data_path=True):
        self.image_list = image_list_  # image_list_ is of pandas's DataFrame format.
        self.base_path = base_path
        self.mode = mode
        self.input_size = input_size
        self.use_base_path = use_base_data_path
        self.train_transform = [
            # T.ToPILImage(),  # this is just for opencv image.
            # T.Resize((self.input_size , self.input_size )),
            T.Resize((self.input_size + self.input_size // 8, self.input_size + self.input_size // 8)),
            T.CenterCrop(self.input_size),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.train_transform = T.Compose(self.train_transform)
        self.test_transform = T.Compose([
            # T.Resize((self.input_size + self.input_size // 8, self.input_size + self.input_size // 8)),
            # T.CenterCrop(self.input_size),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        # read images
        image_flipped, image_, image_name_ = self.read_images(index)
        
        if self.mode in ['train', 'evaluate']:
            label_10 = int(self.image_list.iloc[index].label)
        
        else:  # if not in ['train', 'evaluate'], means the data has no label, so simply return its name.
            label_10 = image_name_
        if self.mode == 'train':
            image_ = self.train_transform(image_)
        else:
            image_ = self.test_transform(image_)
        return image_.float(), label_10,
    
    def read_images(self, index_):
        if self.mode == 'train':
            filename = self.image_list.iloc[index_].filename
        else:
            filename = self.image_list.iloc[index_].filename
        if self.use_base_path:
            image_path_ = os.path.join(self.base_path, filename)
        else:
            image_path_ = filename
        image = Image.open(image_path_).convert('RGB')
        return image, image, filename


if __name__ == "__main__":
    pass

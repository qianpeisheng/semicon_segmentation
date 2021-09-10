import glob
import random
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


class SemiconDataset(Dataset):
    def __init__(self, mode='Train', small_validation=False):
        # Train, valid split 90:10; use tiff data as test
        # Validation images (hand picked): 3_0_0, 8_0_2, 13_0_1

        self.mode = mode
        self.small_validation = small_validation
        if self.mode == 'Train':
            self.images = glob.glob('data/train/images/*')
            # use 0, 255 for visualization
            self.b_masks = glob.glob('data/train/b_masks_255/*')
            self.l_masks = glob.glob('data/train/l_masks_255/*')
            self.s_masks = glob.glob('data/train/s_masks_255/*')
        elif self.mode == 'Validation':
            self.images = glob.glob('data/validation/images/*')
            self.b_masks = glob.glob('data/validation/b_masks_255/*')
            self.l_masks = glob.glob('data/validation/l_masks_255/*')
            self.s_masks = glob.glob('data/validation/s_masks_255/*')
        elif self.mode == 'Test':
            self.segment_size = 2048
            self.images = glob.glob('data/test/images/*')
        else:
            raise ValueError('dataset mode must be Train or Validation')
        self.transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5), inplace=True)]) 

    def __len__(self):
        if self.mode == 'Test':
            num_segments = len(self.images) * 4 * 4
            return num_segments
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.mode == 'Test':
            image_idx = idx // 16
            name = str(self.images[image_idx].split('.')[0].split('/')[-1])
            segment_idx = idx % 16
            image = read_image(self.images[image_idx])
            image = image/255
            segment_row = segment_idx // 4
            segment_column = segment_idx % 4
            segment_image = image[:,2048*segment_row:2048*(segment_row+1),2048*segment_column:2048*(segment_column+1)]
            return self.transform(segment_image), segment_row, segment_column, name

        image = read_image(self.images[idx])
        image = image/255

        b_mask = read_image(self.b_masks[idx])
        l_mask = read_image(self.l_masks[idx])
        s_mask = read_image(self.s_masks[idx])
        b_mask = b_mask/255
        l_mask = l_mask/255
        s_mask = s_mask/255

        if self.mode == 'Train':  # augment data
            # ramdom crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(512, 512))
            image = TF.crop(image, i, j, h, w)
            b_mask = TF.crop(b_mask, i, j, h, w)
            l_mask = TF.crop(l_mask, i, j, h, w)
            s_mask = TF.crop(s_mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                b_mask = TF.hflip(b_mask)
                l_mask = TF.hflip(l_mask)
                s_mask = TF.hflip(s_mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                b_mask = TF.vflip(b_mask)
                l_mask = TF.vflip(l_mask)
                s_mask = TF.vflip(s_mask)

        combined_mask = torch.stack(
            (b_mask, l_mask, s_mask), dim=0).squeeze(1)
        name = str(self.images[idx].split('.')[0].split('/')[-1])
        return self.transform(image), combined_mask, name

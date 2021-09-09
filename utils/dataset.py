import glob
import random
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


class SemiconDataset(Dataset):
    def __init__(self, mode='Train'):
        # Train, valid split 90:10; use tiff data as test
        # Validation images (hand picked): 3_0_0, 8_0_2, 13_0_1

        self.mode = mode
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
        else:
            raise ValueError('dataset mode must be Train or Validation')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
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

        return image, combined_mask, str(self.images[idx].split('.')[0].split('/')[-1])

from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.transforms import transforms


from randaug import RandAugment
from shutil import deepcopy

import cv2

class NewCompose(transforms.Compose):
    def __call__(self, img, factor):
        for t in self.transforms:
            if isinstance(t, RandAugment):
                img = t(img, factor)
            else:
                img = t(img)
        return img

def get_transform(image_size, N, M = 30):
    train_transform = transform.NewCompose([
        '''
        Resize method:
            + transform.Resize((image_size, image_size)),
            + transform.CenterCrop(image_size)
        Consider to use transform.RandomCrop(image_size, pad_if_needed = True) -> don't know how to implement while testing
        '''
        transform.ToTensor(),
        transform.Normalize()
    ])
    val_transform = deepcopy(train_transform)
    train_transform.transform.insert(0, RandAugment(N,M)) # add more augment method in randaug.py
    return (train_transform, val_transform)

class CustomDataset(Dataset):
    def __init__(self, IDs, target, transform, batch_size, train = True):
        self.X = IDs
        self.y = target
        self.transform = transform
        self.bs = batch_size
        self.train = train
        self.count = 0
        
        # Create factor to linear scaling the strength of RandAugment
        self.total_iters = len(self.IDs)//batch_size
        self.iter_idx = 0

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        self.count += 1
        if self.count == batch_size:
            self.iter_idx += 1
            factor = self.iter_idx/self.bs
        image = cv2.imread(self.X[idx])
        image = self.transform(img = image, factor = factor)
        return (image, torch.Tensor([self.y[idx]]).type(torch.LongTensor)) if self.train \
            else image
        
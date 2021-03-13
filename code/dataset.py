from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.transforms import transforms

from randaug import RandAugment
from shutil import deepcopy
import cv2

from utils import get_data_from_df

class NewCompose(transforms.Compose):
    '''
    Add the factor between iter idx and number of iters in 1 epoch
    '''
    def __call__(self, img, factor):
        for t in self.transforms:
            if isinstance(t, RandAugment):
                img = t(img, factor)
            else:
                img = t(img)
        return img

def get_transform(image_size, N, M = 30):
    '''
    This function will apply randaugment as default
    '''
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
    
def get_dataLoader(dataframe, image_size, batch_size, num_workers, shuffle, N, M, drop_last = False):
    '''
    Generate the dataloader for warm up, training and validation mode
    '''
    train, y_train, val, y_val = get_data_from_df(df)
    train_transform, val_transform = get_transform(image_size, N, M)
    log_cache = (
        batch_size,
        image_size,
        shuffle
    )
    train_dataset = CustomDataset(
        image_list = train,
        target = y_train,
        transform = train_transform,
        batch_size = batch_size,
        train = True
    )
    val_dataset = CustomDataset(
        image_list = val,
        target = y_val,
        transform = val_transform,
        batch_size = batch_size,
        train = False
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = drop_last
    )
    val_loader = DataLoader(
        dataset= val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers= num_workers,
        drop_last = False    
    )
    
    # Apply warm up to new layer to make them adapt with the previous weights, normally the new layer will be Dense or Normalization layer
    warm_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size * 5,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = drop_last    
    )
    return train_loader, val_loader, warm_loader, log_cache, len(train)
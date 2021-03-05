import numpy as np
import pandas as pd

from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader

from dataset import ProblemData

def get_data_from_df(df):
    '''
    Fill the necessary extraction
    '''
    return train, y_train, val, y_val

def get_transform(image_size):
    # variable image_size for resize
    train_transform = albumentations.Compose([
        ### Fill the necessary augmentation
        {your_code},
        ###
        albumentations.Normalize(), 
        ToTensorV2()
    ])
    val_transform = albumentations.Compose([
        ### Fill the necessary augmentation
        {your_code},
        ###
        albumentations.Normalize(),
        ToTensorV2() # always use V2 follow this answer: https://albumentations.ai/docs/faq/#which-transformation-should-i-use-to-convert-a-numpy-array-with-an-image-or-a-mask-to-a-pytorch-tensor-totensor-or-totensorv2
    ]
    )
    return train_transform, val_transform

def get_optimizer(opt_type, lr, weight_decay):
    if opt_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr= lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    return optimizer

def get_dataLoader(dataframe, image_size, batch_size, num_workers, shuffle, drop_last = True):
    train, y_train, val, y_val = get_data_from_df(df)
    train_transform, val_transform = get_transform(image_size)
    log_cache = (
        batch_size,
        image_size,
        shuffle
    )
    train_dataset = ProblemData(
        image_list = train,
        target = y_train,
        transform = train_transform
    )
    val_dataset = ProblemData(
        image_list = val,
        target = y_val,
        transform = val_transform
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
    warm_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size * 5,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = drop_last    
    )
    return train_loader, val_loader, warm_loader, log_cache

def grab_df(path):
    import glob
    list_fold = glob.glob('*.csv')
    current_dir = os.getcwd()
    list_path = [current_dir + '/' + i for i in list_fold]
    return [pd.read_csv(i) for i in list_path]

class HiddenPrints: # Block stdout in print function
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def write_log(save_path, model, optimizer, criterion, cache): # dump training information to txt file 
    with HiddenPrints():
        with open(save_path + 'model_info.txt', 'w') as f:
            print(model, file = f)
        with open(save_path + 'training_info.txt', 'w') as f:
            f.write('Optimizer: \n')
            print(optimizer, file = f)
            f.write('Loss Function: \n')
            print(criterion, file = f)

            batch_size, image_size, shuffle, exp = cache
            f.write('Batch size: %d\n'%(batch_size))
            f.write('Input image size: %d\n'%(image_size))
            f.write('Data shuffle: '+str(shuffle) + '\n' )
            f.write('Exponential target: '+str(exp) + '\n' )

def extract_number(path):
    if not os.path.exists(path):
        raise Exception('Path is not exists')
    name, extension = os.path.splitext(path)
    if extension != '.pth':
        raise Exception('Only accept file with pth extension')
    real_name = name.split('/')[-1]
    return int(real_name.split('epoch')[-1])

def fifty_m_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.5

def reset_m_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.running_mean = torch.zeros(m.running_mean.size())
        m.running_var = torch.zeros(m.running_var.size())
        m.momentum = 0.5

def back_normal_m_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.1

def save_pth(path, epoch, model, optimizer, type_opt):
    torch.save({
        'epoch': epoch,
        'pre_opt': type_opt,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
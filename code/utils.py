import numpy as np
import pandas as pd

from torch.optim import Adam, AdamW, SGD
from torch_optimizer import RAdam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn as nn

from dataset import ProblemData

def warm_up(model, dataloader, optimizer, criterion, device):
    # warm up step
    model.train()
    bar = tqdm(dataloader)
    for (image, target) in bar:
        image, target = image.to(device), target.to(device)
        output = model(image)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_np = loss.clone().detach().cpu().item()
        bar.set_description('Loss: %.4f'%(loss_np))

def train_epoch(model, dataloader, optimizer, criterion, evaluation, step_per_snap, num_step, save_path, device):
    model.train()
    bar = tqdm(dataloader)
    all_loss = list()
    snap = 1
    for k, (image, target) in enumerate(bar):
        # init for training
        num_step += 1 # for snapshot
        image, target = image.to(device), target.to(device)
        # predict from model
        output = model(image)
        if k == 0:
            output_numpy = ouput.clone().detach().cpu().numpy().squeeze()
        else:
            temp_numpy = ouput.clone().detach().cpu().numpy().squeeze()
            output_numpy = np.concatenate((output_numpy, temp_numpy), axis = 0)
        # training step
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # visualization
        loss_np = loss.clone().detach().cpu().item()
        all_loss.append(loss_np)
        bar.set_description('Loss: %.4f'%(loss_np))
        # save model when enough step per snap
        if num_step % step_per_snap == 0 and step_per_snap != 1:
            name = save_path + 'snap_%d.pth'%(snap)
            torch.save(model.state_dict(), name)
            snap += 1
        '''
        let edit evaluation for each different dataset
        '''
    epoch_loss = np.mean(np.array(all_loss))
    return epoch_loss, num_step

def val_epoch(model, dataloader, criterion, evaluation, device):
    model.eval()
    bar = tqdm(dataloader)
    all_loss = list()
    with torch.no_grad():
        for k, (image, target) in enumerate(bar):
            image, target = image.to(device), target.to(device)
            # predict from model
            output = model(image)
            if k == 0:
                output_numpy = output.cpu().numpy().squeeze()
            else:
                temp_numpy = ouput.clone().detach().cpu().numpy().squeeze()
                output_numpy = np.concatenate((output_numpy, temp_numpy), axis = 0)
            target = target.numpy()
            # loss fucntion
            loss = criterion(output, target).clone().cpu().item()
            all_loss.append(loss)
            bar.set_description('Loss: %.4f'%(loss))
            '''
            let edit evaluation for each different dataset
            '''
        val_loss = np.mean(np.array(all_loss))
    return val_loss

class SigmoidCrossEntropy(nn.Module):
    def __init__(self):
        super(SigmoidCrossEntropy, self).__init__()
        self.nllloss = nn.NLLLoss()
    def forward(self, x, y):
        return self.nllloss(nn.Sigmoid(x))

def get_data_from_df(df):
    '''
    Fill the necessary extraction
    '''
    return train, y_train, val, y_val

# draft version if the RandAugment is not implemented.
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

def get_optimizer(opt_type, model, lr, weight_decay):
    # get optimizer
    opt_type = opt_type.lower()
    if opt_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr= lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    elif opt_type == 'radam':
        optimizer = RAdam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    elif opt_type == 'adamw':
        optimizer = AdamW(
            model.parameters(), 
            lr = lr
        )
    elif opt_type == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr = lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay
        )
    return optimizer

def grab_df(path):
    # grab all csv file
    # support for get_data_from_df
    import glob
    list_fold = glob.glob('*.csv')
    current_dir = os.getcwd()
    list_path = [current_dir + '/' + i for i in list_fold]
    return [pd.read_csv(i) for i in list_path]

'''
Utilities for writing log function
'''
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

def check_path(dir):
    if os.path.exists(dir):
        print('Save at:', dir)
    else:
        print('No directory, create new directory at:', dir)
        os.makedirs(dir)
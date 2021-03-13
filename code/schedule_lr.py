import math
from torch.optim.lr_scheduler import _LRScheduler

class SnapShotLearningRate(_LRScheduler):
    # implement follow the original paper: https://arxiv.org/abs/1704.00109
    def __init__(self, optimizer, based_lr, step_per_snap, nb_cycles, last_epoch = -1):        
        self.lr = based_lr
        # T = total_epochs * step_per_epochs
        # step_per_snap = T/nb_cycles
        self.step_per_snap = step_per_snap
        self.t = 0
        self.cur_lr = based_lr

        super(SnapShotLearningRate, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        # follow the formular in that paper
        if self.t%self.step_per_snap == 0:
            pass
        else:
            self.cur_lr = self.lr * 0.5 * (math.cos(math.pi*(self.t% self.step_per_snap)/self.step_per_snap) + 1)
        self.t += 1
        return [self.cur_lr for _ in self.base_lrs]

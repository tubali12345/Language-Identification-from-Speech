from datetime import date

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data.audio_preprocess import Aud2Mel
from data.dataloader import ShortAudioDataSet, collate_fn
from models.resnet import ResNet50LangDetection
from train_loop import train_loop


def load_data(train_data_path: str,
              val_data_path: str,
              batch_size: int,
              prefetch_factor: int = 4,
              num_workers: int = 1
              ) -> tuple:
    ds = DataLoader(ShortAudioDataSet(train_data_path, with_augmentation=True), batch_size=batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=False, pin_memory=False, num_workers=num_workers,
                    prefetch_factor=prefetch_factor)
    valid_ds = DataLoader(ShortAudioDataSet(val_data_path, random_sample=False, no_samples_per_class=10 ** 5),
                          batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=False,
                          num_workers=num_workers, prefetch_factor=prefetch_factor)
    return ds, valid_ds


def train(num_epochs: int,
          lr: float,
          max_lr: float,
          batch_size: int,
          num_classes: int,
          out_dir_path: str,
          pct_start: float = 0.1,
          load_epoch: int = 0,
          weights_path: str = None,
          device: str = 'cuda:0'):
    ds, valid_ds = load_data(train_data_path='/home/turib/train_data',
                             val_data_path='/home/turib/train_data_val',
                             batch_size=batch_size)

    model = ResNet50LangDetection(num_classes=num_classes).to(device)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    div_factor = max_lr / 3e-6
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, num_epochs * len(ds),
                                                   div_factor=div_factor,
                                                   pct_start=pct_start,
                                                   final_div_factor=div_factor)
    writer = SummaryWriter(f'runs/{out_dir_path.split("/")[-2]}_{date.today()}')

    train_loop(model=model,
               num_epochs=num_epochs,
               train_ds=ds,
               valid_ds=valid_ds,
               aud_to_mel=Aud2Mel(Config.feature_dim, Config.sample_rate, 2048, 400, 160),
               loss_fn=loss_fn,
               optimizer=optimizer,
               lr_sched=lr_sched,
               writer=writer,
               out_dir_path=out_dir_path,
               load_epoch=load_epoch,
               device=device)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    train(num_epochs=100,
          lr=1e-4,
          max_lr=3e-4,
          batch_size=64,
          num_classes=6,
          out_dir_path=f'/home/turib/lang_detection/weights/ResNet50/weights_03_08',
          device='cuda:0')

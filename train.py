import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, MyDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
dir_origin_train = Path('./data/origin-train/')
dir_train = Path('./data/train/')
dir_test = Path('./data/test/')
dir_checkpoint = Path('./checkpoints/')


def train_net(name,
              net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              checkpoint_epochs=1,
              weight_decay=1e-8,
              data_aug=False,
              loss_func='sum'):
    # 1. Create dataset
    # try:
    #     dataset = MyDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # dataset = MyDataset(dir_img, dir_mask, (args.size, args.size), img_scale)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_set = MyDataset(os.path.join(dir_train if data_aug else dir_origin_train, 'imgs'),
                          os.path.join(dir_train, 'masks'),
                          (args.size, args.size),
                          img_scale)
    # val_set = MyDataset(os.path.join(dir_test, 'imgs'),
    #                     os.path.join(dir_test, 'masks'),
    #                     (args.size, args.size),
    #                     img_scale)
    val_set = train_set
    n_train = len(train_set)
    n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(0), **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment = wandb.init(project='U-Net', resume='allow', entity='tiny_raindrop', name=name)
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Name:               {name}
        Epochs:             {epochs}
        Batch size:         {batch_size}
        Learning rate:      {learning_rate}
        Training size:      {n_train}
        Validation size:    {n_val}
        Checkpoints:        {save_checkpoint}
        Device:             {device.type}
        Images scaling:     {img_scale}
        Mixed Precision:    {amp}
        Weight decay:       {weight_decay}
        Data augmentation:  {data_aug}
        Loss function:      {loss_func}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    if loss_func == 'sum':
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                    elif loss_func == 'ce':
                        loss = criterion(masks_pred, true_masks)
                    elif loss_func == 'dice':
                        loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                         F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                         multiclass=True)
                    else:
                        raise Exception("Loss function can't be recognized.")

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=20)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (10 * batch_size))
                division_step = math.ceil(n_train / batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score_list = evaluate(net, val_loader, device)
                        tumor_val_score = val_score_list[0]
                        muscle_val_score = val_score_list[1]
                        cavity_val_score = val_score_list[2]
                        val_score = (tumor_val_score + muscle_val_score + cavity_val_score) / 3
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Each class: {}'.format(val_score_list))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'tumor validation Dice': tumor_val_score,
                            'muscle validation Dice': muscle_val_score,
                            'cavity validation Dice': cavity_val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint and (epoch + 1) % checkpoint_epochs == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            torch.save(net.state_dict(), str(dir_checkpoint / '{}_epoch{}.pth'.format(args.name, epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--size', '-z', type=int, default=512, help='Length side of images')
    parser.add_argument('--name', '-n', type=str, default='', help='The name of trained model')
    parser.add_argument('--checkpoint_epochs', '-c', type=int, default=10, help='How many epochs to save as checkpoint')
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-8, help='Weight decay factor')
    parser.add_argument('--data_aug', '-da', action='store_true', default=False, help='Use data augmentation')
    parser.add_argument('--loss_func', '-lf', type=str, default='sum',
                        help='Loss function: "ce" for cross entropy, "dice" for DSC, "sum" for sum')
    parser.add_argument('--piles', '-p', type=int, default=5, help='Number of network piles')
    parser.add_argument('--init_feature_channels', '-ifc', type=int, default=64,
                        help="Number of feature's channels in the first layer")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=4, n_piles=args.piles, n_init_feature_channels=args.init_feature_channels,
               bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{net.n_piles} piles\n'
                 f'\t{net.n_init_feature_channels} channels for feature in the first layer\n'
                 f'\t{sum(p.numel() for p in net.parameters())} parameters\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling\n\n'
                 f'{net}')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(name=args.name,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  checkpoint_epochs=args.checkpoint_epochs,
                  weight_decay=args.weight_decay,
                  data_aug=args.data_aug,
                  loss_func=args.loss_func)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

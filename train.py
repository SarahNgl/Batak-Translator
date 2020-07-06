import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.customnet_ssd import create_customnet_ssd
from vision.ssd.config import customnet_ssd_config
from vision.ssd.fpnnet_ssd import create_fpnnet_ssd
from vision.ssd.config import fpnnet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from torchsummary import summary




def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        if type(images) == bool: continue
        images = images.to(device)
        #print(images)
        boxes = boxes.to(device)
        #print(boxes)
        labels = labels.to(device)
        #print(labels)
        
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()
         
       
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
 
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(s)
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            
            


def ssd(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        if type(images) == bool: continue
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)

    create_net = create_fpnnet_ssd
    config = fpnnet_ssd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        dataset = _VOCDataset(dataset_path, transform=train_transform,
                                target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info(" Validation datasets.")

    val_dataset = _VOCDataset(args.validation_dataset, transform=test_transform,
                                target_transform=target_transform, is_test=True)
    logging.info(val_dataset)
    logging.info(" dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = args.last_epoch-1

   
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.45, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1,)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max,)
    elif args.scheduler == 'step':
        logging.info("Uses Step scheduler.")
        scheduler = StepLR(optimizer, step_size=args.step_size,)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    for k in range(last_epoch):
        scheduler.step()


    logging.info(f"Start training from epoch {last_epoch + 1}.")

    summary(net, input_size=(3, config.image_size, config.image_size))
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info()
            model_path = os.path.join(args.checkpoint_folder, f"CNN.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")

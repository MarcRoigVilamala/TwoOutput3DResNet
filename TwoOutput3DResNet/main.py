import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from TwoOutput3DResNet.opts import parse_opts
from TwoOutput3DResNet.model import generate_model
from TwoOutput3DResNet.Transformations.spatial_transforms import get_spatial_transform, get_norm_method
from TwoOutput3DResNet.Transformations.temporal_transforms import LoopPadding, TemporalRandomCrop
from TwoOutput3DResNet.Transformations.target_transforms import TimeStampLabel, VideoIDAndFrames
from TwoOutput3DResNet.dataset import get_training_set, get_validation_set, get_test_set
from TwoOutput3DResNet.utils import Logger
from TwoOutput3DResNet.train import train_epoch
from TwoOutput3DResNet.validation import val_epoch
from TwoOutput3DResNet import test

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    norm_method = get_norm_method(opt)

    if not opt.no_train:
        spatial_transform = get_spatial_transform(opt, norm_method, 'train')
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        # target_transform = ClassLabel()
        target_transform = TimeStampLabel('Temporal_Anomaly_Annotation_for_Testing_Videos.txt')
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = get_spatial_transform(opt, norm_method, 'val')
        temporal_transform = LoopPadding(opt.sample_duration)
        # target_transform = ClassLabel()
        target_transform = TimeStampLabel('Temporal_Anomaly_Annotation_for_Testing_Videos.txt')
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        spatial_transform = get_spatial_transform(opt, norm_method, 'test')
        temporal_transform = LoopPadding(opt.sample_duration)
        # target_transform = VideoID()
        target_transform = VideoIDAndFrames()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        # test.test(test_loader, model, opt, test_data.class_names)
        test.every_segment_test(test_loader, model, opt, test_data.class_names)

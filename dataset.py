from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.ucfcrime import UCFCRIME


DATASET_CLASS = {
    'kinetics': Kinetics,
    'activitynet':
        lambda untrimmed: lambda *args, **kwargs: ActivityNet(is_untrimmed_setting=untrimmed, *args, **kwargs),
    'ucf101': UCF101,
    'UCF_CRIME': UCFCRIME,
    'hmdb51': HMDB51,
}


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'UCF_CRIME']

    dataset = DATASET_CLASS[opt.dataset]

    if opt.dataset == 'activitynet':
        dataset = dataset(False)

    training_data = dataset(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform
    )

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'UCF_CRIME']

    dataset = DATASET_CLASS[opt.dataset]

    if opt.dataset == 'activitynet':
        dataset = dataset(False)

    validation_data = dataset(
        opt.video_path,
        opt.annotation_path,
        'validation',
        n_samples_for_each_video=opt.n_val_samples,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration
    )

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'UCF_CRIME']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'

    dataset = DATASET_CLASS[opt.dataset]

    if opt.dataset == 'activitynet':
        dataset = dataset(True)

    test_data = dataset(
        opt.video_path,
        opt.annotation_path,
        subset,
        n_samples_for_each_video=0,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration
    )

    return test_data

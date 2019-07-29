import sys

import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

from TwoOutput3DResNet.utils import load_value_file


class UCF101(data.Dataset):
    @classmethod
    def pil_loader(cls, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    @classmethod
    def accimage_loader(cls, path):
        try:
            import accimage
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return UCF101.pil_loader(path)

    @classmethod
    def get_default_image_loader(cls):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return cls.accimage_loader
        else:
            return cls.pil_loader

    @classmethod
    def video_loader(cls, video_dir_path, frame_indices, image_loader):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                return video

        return video

    @classmethod
    def get_default_video_loader(cls):
        image_loader = cls.get_default_image_loader()
        return functools.partial(cls.video_loader, image_loader=image_loader)

    @classmethod
    def load_annotation_data(cls, data_file_path):
        with open(data_file_path, 'r') as data_file:
            return json.load(data_file)

    @classmethod
    def get_class_labels(cls, data):
        class_labels_map = {}
        index = 0
        for class_label in data['labels']:
            class_labels_map[class_label] = index
            index += 1
        return class_labels_map

    @classmethod
    def get_video_names_and_annotations(cls, data, subset):
        video_names = []
        annotations = []

        for key, value in data['database'].items():
            this_subset = value['subset']
            if this_subset == subset:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

        return video_names, annotations

    @classmethod
    def get_videos(cls, root_path, video_names):
        for i, video_name in enumerate(video_names):
            if i % 1000 == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_names)))

            video_path = os.path.join(root_path, video_name)
            if not os.path.exists(video_path):
                print('{} not found!'.format(video_path), file=sys.stderr)
                continue

            n_frames_file_path = os.path.join(video_path, 'n_frames')
            n_frames = int(load_value_file(n_frames_file_path))
            if n_frames <= 0:
                print('{} does not have frames!'.format(video_path), file=sys.stderr)
                continue

            yield i, video_path, video_name, n_frames

    @classmethod
    def get_samples(cls, video_path, video_name, n_frames, label, n_samples_for_each_video, sample_duration):
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_name.split('/')[1],
            'label': label
        }


        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            return [sample]
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration

            res = []
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                res.append(sample_j)

            return res

    @classmethod
    def sub_make_dataset(cls, root_path, video_names, annotations, class_to_idx,
                         n_samples_for_each_video, sample_duration):
        dataset = []
        for i, video_path, video_name, n_frames in cls.get_videos(root_path, video_names):
            if len(annotations) != 0:
                label = class_to_idx[annotations[i]['label']]
            else:
                label = -1

            dataset += cls.get_samples(
                video_path, video_name, n_frames, label, n_samples_for_each_video, sample_duration
            )

        return dataset

    @classmethod
    def make_dataset(cls, root_path, annotation_path, subset, n_samples_for_each_video,
                     sample_duration):
        data = cls.load_annotation_data(annotation_path)
        video_names, annotations = cls.get_video_names_and_annotations(data, subset)
        class_to_idx = cls.get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        return cls.sub_make_dataset(
            root_path, video_names, annotations, class_to_idx, n_samples_for_each_video, sample_duration
        ), idx_to_class

    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=None):
        self.data, self.class_names = self.make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        if get_loader is None:
            get_loader = self.get_default_video_loader
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        video_data = dict(self.data[index])
        path = video_data['video']

        frame_indices = video_data['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            video_data['frame_indices'] = frame_indices
            video_data['segment'] = [min(frame_indices), max(frame_indices)]
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            target = self.target_transform(video_data)

        return clip, target

    def __len__(self):
        return len(self.data)

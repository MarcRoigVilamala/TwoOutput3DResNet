import os
import sys

from TwoOutput3DResNet.datasets.ucf101 import UCF101
from TwoOutput3DResNet.utils import load_value_file


class SingleEvaluation(UCF101):
    @classmethod
    def get_videos(cls, root_path, video_names):
        video_name = '/'.join(root_path.split('/')[-2:])
        if not os.path.exists(root_path):
            print('{} not found!'.format(root_path), file=sys.stderr)
        else:
            n_frames_file_path = os.path.join(root_path, 'n_frames')
            n_frames = int(load_value_file(n_frames_file_path))
            if n_frames <= 0:
                print('{} does not have frames!'.format(root_path), file=sys.stderr)
            else:
                yield 0, root_path, video_name, n_frames

    @classmethod
    def make_dataset(cls, root_path, annotation_path, subset, n_samples_for_each_video,
                     sample_duration):
        video_names = ['']
        annotations = []
        class_to_idx = {}
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        return cls.sub_make_dataset(
            root_path, video_names, annotations, class_to_idx, n_samples_for_each_video, sample_duration
        ), idx_to_class

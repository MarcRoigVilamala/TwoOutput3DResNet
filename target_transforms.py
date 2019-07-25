import random
import math
import sys

import pandas as pd


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


def overlap(inc, seg):
    return inc[1] >= seg[0] and seg[1] >= inc[0]


class TimeStampLabel(object):
    def __init__(self, annotations_file, timestamp_check=overlap, classes=(0, 1)):
        temp_annot = pd.read_csv(
            annotations_file,
            delimiter='  ',
            header=None,
            names=['filename', 'video_type', 'start1', 'end1', 'start2', 'end2'],
            engine='python'
        )

        self.timestamp_check = timestamp_check
        self.classes = classes

        self.incident_timestamps = {}
        for index, row in temp_annot.iterrows():
            incidents = []

            if row['start1'] != -1:
                incidents.append(
                    (row['start1'], row['end1'])
                )

            if row['start2'] != -1:
                incidents.append(
                    (row['start2'], row['end2'])
                )

            self.incident_timestamps[row['filename'].split('.')[0]] = incidents

    def __call__(self, target):
        incidents = self.incident_timestamps[target['video_id']]

        segment = target['segment']

        for inc in incidents:
            if self.timestamp_check(inc, segment):
                return self.classes[1]

        return self.classes[0]


class ClassLabel(object):

    def __call__(self, target):
        return target['label']


class VideoID(object):

    def __call__(self, target):
        return target['video_id']

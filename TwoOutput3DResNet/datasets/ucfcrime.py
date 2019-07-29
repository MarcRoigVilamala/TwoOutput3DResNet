import re
from TwoOutput3DResNet.datasets.ucf101 import *


class UCFCRIME(UCF101):
    @classmethod
    def get_video_names_and_annotations(cls, data, subset):
        video_names = []
        annotations = []

        for key, value in data['database'].items():
            this_subset = value['subset']
            if this_subset == subset:
                folder = re.match(r'(\w*)\d{3}_x264', key).group(1)

                # if folder == 'Normal_Videos_':
                #     folder = '../Normal_Videos_for_Event_Recognition'

                video_names.append('{}/{}'.format(folder, key))
                annotations.append(value['annotations'])

        return video_names, annotations


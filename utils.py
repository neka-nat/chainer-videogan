import os
import cv2
import numpy as np

class DataLoader(object):
    def __init__(self, root_path, batch_size=64):
        self._root_path = root_path
        self._batch_size = batch_size
        self._frame_size = 32
        self._crop_size = 64
        self._image_size = 128
        self._data_list_path = os.path.join(self._root_path, 'golf.txt')
        with open(self._data_list_path, 'r') as f:
            self._video_index = [x.strip() for x in f.readlines()]
            np.random.shuffle(self._video_index)
        self._cursor = 0

    def get_batch(self):
        if self._cursor + self._batch_size > len(self._video_index):
            self._cursor = 0
            np.random.shuffle(self._video_index)
        out = np.zeros((self._batch_size, self._frame_size, 3, self._crop_size, self._crop_size), dtype=np.float32)
        for idx in xrange(self._batch_size):
            video_path = os.path.join(self._root_path, self._video_index[self._cursor])
            self._cursor += 1
            inputimage = cv2.imread(video_path)
            count = inputimage.shape[0] / self._image_size
            for j in xrange(self._frame_size):
                if j < count:
                    cut = j * self._image_size
                else:
                    cut = (count - 1) * self._image_size
                crop = inputimage[cut : cut + self._image_size, :]
                out[idx, j, :, :, :] = np.transpose(cv2.resize(crop, (self._crop_size, self._crop_size)), (2, 0, 1))

        out = np.transpose(out, (0, 2, 1, 3, 4))
        out = (out - 128.0) / 128.0
        return out
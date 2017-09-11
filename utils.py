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

class VideoLoader(object):
    def __init__(self, root_path, batch_size=64):
        self._batch_size = batch_size
        self._frame_size = 32
        self._crop_size = 64
        with open(os.path.join(root_path, 'train_list.txt'), 'r') as f:
            self._data = []
            for filename in f.readlines():
                vdata = []
                filename = filename.strip()
                vidcap = cv2.VideoCapture(os.path.join(root_path, filename))
                while True:
                    success, image = vidcap.read()
                    if not success:
                        break
                    vdata.append(np.transpose(cv2.resize(image,
                                                         (self._crop_size, self._crop_size)),
                                              (2, 0, 1)))
                self._data.append(np.array(vdata, dtype=np.float32))
            print("Load %d videos" % len(self._data))

    def get_batch(self):
        out = np.zeros((self._batch_size, self._frame_size, 3, self._crop_size, self._crop_size), dtype=np.float32)
        for idx in xrange(self._batch_size):
            vid = np.random.randint(0, len(self._data))
            start_idx = np.random.randint(0, len(self._data[vid]) - self._frame_size)
            for j in xrange(self._frame_size):
                out[idx, j, :, :, :] = self._data[vid][start_idx + j, :, :, :]
        out = np.transpose(out, (0, 2, 1, 3, 4))
        out = (out - 128.0) / 128.0
        return out

if __name__ == "__main__":
    def clip_img(x):
        return np.float32(max(min(1, x), -1))
    loader = VideoLoader("data/", 1)
    print len(loader._data)
    for i in range(10):
        tmp = ((np.vectorize(clip_img)(loader.get_batch()[0, :, :, :, :]) + 1) / 2).transpose(1, 2, 3, 0)
        tmp = np.concatenate(tmp)
        cv2.imwrite('vis_%d.png' % i, tmp * 255.0)

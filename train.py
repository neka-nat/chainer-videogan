from __future__ import print_function
import os
import argparse
import chainer
from chainer import Variable
from chainer import optimizers
from chainer import serializers
from chainer import cuda
from chainer import functions as F
import numpy as np
import cv2
from model import Generator, Discriminator, Predictor
from utils import DataLoader, VideoLoader

parser = argparse.ArgumentParser(description='Train video-gan.')
parser.add_argument('--data_dir', '-d', type=str, default='./data', help='Data directory.')
parser.add_argument('--gpu_no', '-g', type=int, default=0, help='GPU device no.')
parser.add_argument('--predict_model', '-p', action='store_true', default=False, help='Prediction model.')
parser.add_argument('--video_data', '-v', action='store_true', default=False, help='Use video data.')
args = parser.parse_args()

nz = 100 # of dim for Z
batchsize = 8
n_epoch = 100
n_train = 100000
save_interval = 20000
result_dir = './result'
model_dir = './model'
frame_size = 32
xp = cuda.cupy
cuda.get_device(args.gpu_no).use()
if args.video_data:
    loader = VideoLoader(args.data_dir, batchsize)
else:
    loader = DataLoader(args.data_dir, batchsize)

def clip_img(x):
    return np.float32(max(min(1, x), -1))

def train(gen, dis, epoch0=0, predict_model=False):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    lmd = 0.01
    vissize = 16
    zvis = (xp.random.uniform(-1, 1, (vissize, nz), dtype=np.float32))
    
    for epoch in xrange(epoch0, n_epoch):
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        sum_mse = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x2 = loader.get_batch()

            # train generator
            if predict_model:
                z = Variable(cuda.to_gpu(x2[:, :, 0, :, :]))
            else:
                z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)
            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            if predict_model:
                mse = F.mean_squared_error(x[:, :, 0, :, :], z)
                L_gen += lmd * mse
                sum_mse += mse.data.get()
            L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))

            # train discriminator
            x2 = Variable(cuda.to_gpu(x2))
            yl2 = dis(x2)
            L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()

            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()

            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()

            if i % save_interval==0:
                if predict_model:
                    xtest = loader.get_batch()
                    z = cuda.to_gpu(xtest[:, :, 0, :, :])
                else:
                    z = zvis
                    z[8:, :] = (xp.random.uniform(-1, 1, (8, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data.get()
                for j in range(z.shape[0]):
                    if predict_model:
                        in_img = ((xtest[j, :, 0, :, :] + 1) / 2).transpose(1, 2, 0)
                        cv2.imwrite('%s/initial_%d_%d_%d.png' % (result_dir, epoch, i, j), in_img * 255.0)
                    tmp = ((np.vectorize(clip_img)(x[j, :, :, :, :]) + 1) / 2).transpose(1, 2, 3, 0)
                    tmp = np.concatenate(tmp)
                    cv2.imwrite('%s/vis_%d_%d_%d.png' % (result_dir, epoch, i, j), tmp * 255.0)
                
                serializers.save_hdf5("%s/model_dis_%d.h5" % (model_dir, epoch), dis)
                serializers.save_hdf5("%s/model_gen_%d.h5" % (model_dir, epoch), gen)
                serializers.save_hdf5("%s/state_dis_%d.h5" % (model_dir, epoch), o_dis)
                serializers.save_hdf5("%s/state_gen_%d.h5" % (model_dir, epoch), o_gen)
        print('epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train, sum_mse/n_train)

if args.predict_model:
    gen = Predictor()
else:
    gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()

try:
    os.mkdir(result_dir)
    os.mkdir(model_dir)
except:
    pass

train(gen, dis, predict_model=args.predict_model)

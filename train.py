from __future__ import print_function
import argparse
import chainer
from chainer import optimizers
from chainer import cuda
import numpy as np
from model import Generator, Discriminator
from utils import DataLoader

parser = argparse.ArgumentParser(description='Train video-gan.')
parser.add_argument('--data_dir', '-d', type=str, default='.', help='Data directory.')
args = parser.parse_args()

nz = 100 # of dim for Z
batchsize = 64
n_epoch = 10000
n_train = 200000
save_interval = 50000
result_dir = './result'
model_dir = './model'
frame_size = 32
xp = cuda.cupy
cuda.get_device(0).use()
loader = DataLoader(args.data_dir, batchsize)

def clip_img(x):
    return np.float32(max(min(1, x), -1))

def train(gen, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    vissize = 16
    zvis = (xp.random.uniform(-1, 1, (vissize, nz), dtype=np.float32))
    
    for epoch in xrange(epoch0, n_epoch):
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x2 = loader.get_batch()

            # train generator
            z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)
            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
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
                z = zvis
                z[8:, :] = (xp.random.uniform(-1, 1, (8, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data.get()
                for f in range(frame_size):
                    pylab.rcParams['figure.figsize'] = (16.0, 16.0)
                    pylab.clf()
                    for j in range(vissize):
                        tmp = ((np.vectorize(clip_img)(x[j, f, :, :, :]) + 1) / 2).transpose(1, 2, 0)
                        pylab.subplot(4, 4, j + 1)
                        pylab.imshow(tmp)
                        pylab.axis('off')
                    pylab.savefig('%s/vis_%d_%d_%d.png' % (result_dir, epoch, i, f))
                
                serializers.save_hdf5("%s/model_dis_%d.h5" % (model_dir, epoch), dis)
                serializers.save_hdf5("%s/model_gen_%d.h5" % (model_dir, epoch), gen)
                serializers.save_hdf5("%s/state_dis_%d.h5" % (model_dir, epoch), o_dis)
                serializers.save_hdf5("%s/state_gen_%d.h5" % (model_dir, epoch), o_gen)
        print('epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train)

gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()

train(gen, dis)

import chainer
import chainer.functions as F
import chainer.links as L
from tb_chainer import name_scope, within_name_scope

_tol = lambda it: [x for x in it]

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            # for background
            bn0b = L.BatchNormalization(512),
            bn1b = L.BatchNormalization(256),
            bn2b = L.BatchNormalization(128),
            bn3b = L.BatchNormalization(64),
            dn0b = L.Deconvolution2D(100, 512, 4),
            dn1b = L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            dn2b = L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dn3b = L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dn4b = L.Deconvolution2D(64, 3, 4, stride=2, pad=1),
            # for foreground
            bn0f = L.BatchNormalization(512),
            bn1f = L.BatchNormalization(256),
            bn2f = L.BatchNormalization(128),
            bn3f = L.BatchNormalization(64),
            dn0f = L.DeconvolutionND(3, 100, 512, (2,4,4)),
            dn1f = L.DeconvolutionND(3, 512, 256, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            dn2f = L.DeconvolutionND(3, 256, 128, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            dn3f = L.DeconvolutionND(3, 128, 64, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            dn4f = L.DeconvolutionND(3, 64, 3, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            dnm = L.DeconvolutionND(3, 64, 1, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
        )

    def background(self, z):
        with name_scope("background",
                        _tol(self.bn0b.params()) + _tol(self.bn1b.params()) + \
                        _tol(self.bn2b.params()) + _tol(self.bn3b.params()) + \
                        _tol(self.dn0b.params()) + _tol(self.dn1b.params()) + \
                        _tol(self.dn2b.params()) + _tol(self.dn3b.params()) + \
                        _tol(self.dn4b.params()),
                        True):
            zin = F.reshape(z, (-1, 100, 1, 1))
            h = F.relu(self.bn0b(self.dn0b(zin)))
            h = F.relu(self.bn1b(self.dn1b(h)))
            h = F.relu(self.bn2b(self.dn2b(h)))
            h = F.relu(self.bn3b(self.dn3b(h)))
            x = F.tanh(self.dn4b(h))
        return x

    def foreground(self, z):
        with name_scope("foreground",
                        _tol(self.bn0f.params()) + _tol(self.bn1f.params()) + \
                        _tol(self.bn2f.params()) + _tol(self.bn3f.params()) + \
                        _tol(self.dn0f.params()) + _tol(self.dn1f.params()) + \
                        _tol(self.dn2f.params()) + _tol(self.dn3f.params()) + \
                        _tol(self.dn4f.params()),
                        True):
            zin = F.reshape(z, (-1, 100, 1, 1, 1))
            h = F.relu(self.bn0f(self.dn0f(zin)))
            h = F.relu(self.bn1f(self.dn1f(h)))
            h = F.relu(self.bn2f(self.dn2f(h)))
            h = F.relu(self.bn3f(self.dn3f(h)))
            mask = self.dnm(h)
            mask = F.sigmoid(mask)
            x = F.tanh(self.dn4f(h))
        return x, mask

    @within_name_scope("Generator", True)
    def __call__(self, z):
        gf, mask = self.foreground(z)
        mask = F.tile(mask, (1, 3, 1, 1, 1))
        gb = self.background(z)
        gb = F.expand_dims(gb, 2)
        gb = F.tile(gb, (1, 1, 32, 1, 1))
        return mask * gf + (1 - mask) * gb

class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__(
            bn0 = L.BatchNormalization(128, eps=1e-3),
            bn1 = L.BatchNormalization(256, eps=1e-3),
            bn2 = L.BatchNormalization(512, eps=1e-3),
            cn0 = L.Convolution2D(3, 64, (4,4), stride=(2,2), pad=(1,1)),
            cn1 = L.Convolution2D(64, 128, (4,4), stride=(2,2), pad=(1,1)),
            cn2 = L.Convolution2D(128, 256, (4,4), stride=(2,2), pad=(1,1)),
            cn3 = L.Convolution2D(256, 512, (4,4), stride=(2,2), pad=(1,1)),
            cn4 = L.Convolution2D(512, 100, (4,4), stride=(1,1), pad=(0,0)),
        )

    @within_name_scope("Encoder", True)
    def __call__(self, x):
        h = F.leaky_relu(self.cn0(x), 0.2)
        h = F.leaky_relu(self.bn0(self.cn1(h)), 0.2)
        h = F.leaky_relu(self.bn1(self.cn2(h)), 0.2)
        h = F.leaky_relu(self.bn2(self.cn3(h)), 0.2)
        return F.reshape(self.cn4(h), (-1, 100))

class Predictor(chainer.Chain):
    def __init__(self):
        super(Predictor, self).__init__(
            enc = Encoder(),
            gen = Generator(),
        )
    def __call__(self, x):
        z = self.enc(x)
        return self.gen(z)

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            bn0 = L.BatchNormalization(128, eps=1e-3),
            bn1 = L.BatchNormalization(256, eps=1e-3),
            bn2 = L.BatchNormalization(512, eps=1e-3),
            cn0 = L.ConvolutionND(3, 3, 64, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            cn1 = L.ConvolutionND(3, 64, 128, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            cn2 = L.ConvolutionND(3, 128, 256, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            cn3 = L.ConvolutionND(3, 256, 512, (4,4,4), stride=(2,2,2), pad=(1,1,1)),
            cn4 = L.ConvolutionND(3, 512, 2, (2,4,4), stride=(1,1,1), pad=(0,0,0)),
        )

    @within_name_scope("Discriminator", True)
    def __call__(self, x):
        h = F.leaky_relu(self.cn0(x), 0.2)
        h = F.leaky_relu(self.bn0(self.cn1(h)), 0.2)
        h = F.leaky_relu(self.bn1(self.cn2(h)), 0.2)
        h = F.leaky_relu(self.bn2(self.cn3(h)), 0.2)
        return F.reshape(self.cn4(h), (-1, 2))

if __name__ == "__main__":
    import chainer.computational_graph as c
    import numpy as np
    gen_model = Generator()
    z = chainer.Variable(np.zeros((1, 100), dtype=np.float32))
    g = c.build_computational_graph(gen_model(z))
    with open('gen_network.dot', 'w') as o:
        o.write(g.dump())

    enc_model = Encoder()
    x0 = chainer.Variable(np.zeros((1, 3, 64, 64), dtype=np.float32))
    e = c.build_computational_graph(enc_model(x0))
    with open('encoder_network.dot', 'w') as o:
        o.write(e.dump())

    dc_model = Discriminator()
    x = chainer.Variable(np.zeros((1, 3, 32, 64, 64), dtype=np.float32))
    d = c.build_computational_graph(dc_model(x))
    with open('dc_network.dot', 'w') as o:
        o.write(d.dump())

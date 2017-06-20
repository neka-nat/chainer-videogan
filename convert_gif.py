import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert to gif file.')
parser.add_argument('image_file', type=str, help='Image file.')
parser.add_argument('--out_file', '-o', type=str, default='out.gif', help='Output file.')
args = parser.parse_args()
img = mpimg.imread(args.image_file)
n_frame = img.shape[0] / img.shape[1]
imgs = np.split(img, n_frame)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.spines['left'].set_color('None')
ax.spines['bottom'].set_color('None')
ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
ani = animation.ArtistAnimation(fig,
                                [[plt.imshow(im, interpolation="spline36")] for im in imgs],
                                interval=200)
ani.save(args.out_file, writer='imagemagick')

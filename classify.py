#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import time
import webbrowser

caffe_root = './'
import sys
sys.path.insert(0, caffe_root + 'python')

# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors
import os
os.environ['GLOG_minloglevel'] = '0' 
import caffe

net = sys.argv[1]
img = sys.argv[2]

# select the net
if net == '-caffenet':
	net_fn   = 'models/bvlc_reference_caffenet/deploy.prototxt'
	param_fn = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	height = width = 227
elif net == '-alexnet':
	net_fn   = 'models/bvlc_alexnet/deploy.prototxt'
	param_fn = 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
	height = width = 227
elif net == '-googlenet':
	net_fn = 'models/bvlc_googlenet/deploy.prototxt'
	param_fn = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	height = width = 224
elif net == '-flickrnet':
	net_fn   = 'models/finetune_flickr_style/deploy.prototxt'
	param_fn = 'models/finetune_flickr_style/finetune_flickr_style.caffemodel'
	height = width = 227
else:
	net_fn   = 'models/bvlc_reference_caffenet/deploy.prototxt'
	param_fn = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	height = width = 227

# set caffe to GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

# load the net in the test phase for inference
net = caffe.Net(
	caffe_root + net_fn,
	caffe_root + param_fn,
	caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,height,width)

# feed in the image (with some preprocessing) and classify with a forward pass
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
start = time.time()
out = net.forward()
print('\nDone in %.5f s.' % (time.time() - start))
print('\nPredicted class is #{}.'.format(out['prob'][0].argmax()))

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
for i in range(len(top_k)):
	print str(i+1) + ': ' + labels[top_k[i]] + ' (%.2f' % (out['prob'][0][top_k[i]]*100) + '%)'
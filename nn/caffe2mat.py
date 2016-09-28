import caffe
from caffe.proto import caffe_pb2
import sys
import os
import numpy as np

model_deploy_file = '/orions3-zfs/projects/rqi/Code/3dcnn/experiments/occupancy_30_5x_tilt_elevation_aug_modelnet40/nin_eight_regions/deploy.prototxt'
model_params_file = '/orions3-zfs/projects/rqi/Code/3dcnn/experiments/occupancy_30_5x_tilt_elevation_aug_modelnet40/nin_eight_regions/snapshots_iter_30000.caffemodel'


net = caffe.Net(model_deploy_file, model_params_file, caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
raw_input()

conv1w = net.params['conv1'][0].data[...]
conv1b = net.params['conv1'][1].data[...]
print np.shape(conv1w)
print np.shape(conv1b)
cccp11w = net.params['cccp11'][0].data[...]
cccp11b = net.params['cccp11'][1].data[...]
cccp12w = net.params['cccp12'][0].data[...]
cccp12b = net.params['cccp12'][1].data[...]

conv2w = net.params['conv2'][0].data[...]
conv2b = net.params['conv2'][1].data[...]
cccp21w = net.params['cccp21'][0].data[...]
cccp21b = net.params['cccp21'][1].data[...]
cccp22w = net.params['cccp22'][0].data[...]
cccp22b = net.params['cccp22'][1].data[...]

conv3w = net.params['conv3'][0].data[...]
conv3b = net.params['conv3'][1].data[...]
cccp31w = net.params['cccp31'][0].data[...]
cccp31b = net.params['cccp31'][1].data[...]
cccp32w = net.params['cccp32'][0].data[...]
cccp32b = net.params['cccp32'][1].data[...]

fc4w = net.params['fc4'][0].data[...]
fc4b = net.params['fc4'][1].data[...]

fc5w = net.params['fc5'][0].data[...]
fc5b = net.params['fc5'][1].data[...]

fc6w = net.params['fc6'][0].data[...]
fc6b = net.params['fc6'][1].data[...]

import scipy.io as sio
sio.savemat('subvolumesup_param.mat',{
'conv1b':conv1b, 'conv1w':conv1w, 'cccp11b':cccp11b, 'cccp11w':cccp11w, 'cccp12b':cccp12b,'cccp12w':cccp12w,
'conv2b':conv2b, 'conv2w':conv2w, 'cccp21b':cccp21b, 'cccp21w':cccp21w, 'cccp22b':cccp22b,'cccp22w':cccp22w,
'conv3b':conv3b, 'conv3w':conv3w, 'cccp31b':cccp31b, 'cccp31w':cccp31w, 'cccp32b':cccp32b,'cccp32w':cccp32w,
'fc4b':fc4b, 'fc4w':fc4w, 'fc5b':fc5b, 'fc5w':fc5w, 'fc6b':fc6b, 'fc6w':fc6w})

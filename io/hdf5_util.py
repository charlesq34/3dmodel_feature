import os
import sys
import numpy as np
import datetime
import scipy.io as sio
import h5py
'''
    An example showing how to write .h5 file in python

    Author: Charles QI
    Last updated: Feb 18, 2016
'''

def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
    )
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype,
    )
    h5_fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    # f.keys() should be [u'data', u'label']
    data = f['data'][:]
    label = f['label'][:]
    return (data,label)


if __name__ == "__main__":
    # =============================================================================
    # Specify what data and label to write, CHNAGE this according to your needs...
    N = 1000
    data_dim = [2,2,2,3,3]
    label_dim = [1]
    data_dtype = 'float32' # or 'uint8'
    label_dtype = 'uint8'
    tensor_data = [np.random.random(tuple(data_dim)) for _ in range(N)]
    tensor_label = [np.zeros(label_dim) for _ in range(N)]
    output_filename_prefix = 'unit_test'
    # =============================================================================
    
    # Note: Caffe has limit on how many instance can be put to a single .h5 file
    h5_batch_size = min(2**31 / np.prod(data_dim) - 1, 10000)
    
    # set batch buffer
    batch_data_dim = [min(h5_batch_size,N)] + data_dim
    batch_label_dim = [min(h5_batch_size,N)] + label_dim
    h5_batch_data = np.zeros(batch_data_dim)
    h5_batch_label = np.zeros(batch_label_dim)
    
    for k in range(N):
        d = tensor_data[k]
        l = tensor_label[k]
        h5_batch_data[k%h5_batch_size, ...] = d
        h5_batch_label[k%h5_batch_size, ...] = l
        
        if (k+1)%h5_batch_size == 0 or k==N-1:
            print '[%s] %d/%d' % (datetime.datetime.now(), k+1, N)
            print 'batch data shape: ', h5_batch_data.shape
            h5_filename = output_filename_prefix+str(k/h5_batch_size)+'.h5'
            begidx = 0
            endidx = min(h5_batch_size, (k%h5_batch_size)+1) 
            save_h5(h5_filename, h5_batch_data[begidx:endidx,:,:,:,:], h5_batch_label[begidx:endidx,:], data_dtype, label_dtype) 
    
    
    # ============================================
    # Verify raw data and loaded data are the same
    (d,l) = load_h5(output_filename_prefix+'0.h5')
    print "Data difference (should be close to 0): ", np.linalg.norm(d - tensor_data)
    print "Label difference (should be close to 0): ", np.linalg.norm(l - tensor_label)

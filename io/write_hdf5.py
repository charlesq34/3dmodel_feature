from hdf5_util import *

output_filename_prefix = 'volume_data'

def write_tensor_label_hdf5(mat_filelist, volume_size, pad_size):
    """ We will use constant label (class 0) for the test data """
    tensor_filenames = [line.rstrip() for line in open(mat_filelist, 'r')]
    labels = [0 for _ in range(len(tensor_filenames))]

    N = len(tensor_filenames)
    assert(N<=10000)

    # =============================================================================
    # Specify what data and label to write, CHNAGE this according to your needs...
    vox_size = volume_size + pad_size * 2
    data_dim = [1,vox_size, vox_size, vox_size]
    label_dim = [1]
    data_dtype = 'uint8'
    label_dtype = 'uint8'
    # =============================================================================
    
    h5_batch_size = N
    
    # set batch buffer
    batch_data_dim = [min(h5_batch_size,N)] + data_dim
    batch_label_dim = [min(h5_batch_size,N)] + label_dim
    h5_batch_data = np.zeros(batch_data_dim)
    h5_batch_label = np.zeros(batch_label_dim)
    
    for k in range(N):
        mat = sio.loadmat(tensor_filenames[k])
        d = mat[mat.keys()[0]]
        l = labels[k]

        h5_batch_data[k%h5_batch_size, ...] = d
        h5_batch_label[k%h5_batch_size, ...] = l
        
        if (k+1)%h5_batch_size == 0 or k==N-1:
            print '[%s] %d/%d' % (datetime.datetime.now(), k+1, N)
            print 'batch data shape: ', h5_batch_data.shape
            h5_filename = output_filename_prefix+str(k/h5_batch_size)+'.h5'
            print h5_filename
            print np.shape(h5_batch_data)
            print np.shape(h5_batch_label)
            begidx = 0
            endidx = min(h5_batch_size, (k%h5_batch_size)+1) 
            print h5_filename, data_dtype, label_dtype
            save_h5(h5_filename, h5_batch_data[begidx:endidx,:,:,:,:], h5_batch_label[begidx:endidx,:], data_dtype, label_dtype) 


write_tensor_label_hdf5('mat_filelist.txt', 26, 3)
(d,l) = load_h5(output_filename_prefix+'0.h5')
print d.shape
print l.shape

import numpy as np
from scipy import spatial

model_names = [x.rstrip().split('/')[-1] for x in open('io/obj_filelist.txt')]

# Load features, each row is a feature
feats = np.loadtxt('output.txt', delimiter=',')

# Compute distances..
dists = spatial.distance.pdist(feats)
dists = spatial.distance.squareform(dists)

# Get nearest neighbor for each obj
for i in range(dists.shape[0]):
    neighbor_idx = np.argsort(dists[i, :])[1]
    print "Model %s's nearest neighbor is %s with distance %f" \
            % (model_names[i], model_names[neighbor_idx], dists[i, neighbor_idx])

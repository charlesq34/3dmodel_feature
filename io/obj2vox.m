function volume = obj2vox(obj_filename, volume_size, pad_size, visu)
% OBJ2VOX, convert an obj model to binary volume
% Input:
%   obj_filename: string, obj model file path. The OBJ model will be preprocessed
%       with translation to positive, scaling and offset translation to center
%       the preprocessing makes the convertion translation-invariant and
%       scale-invariant to the input OBJ model.
%   volume_size: integer, final volume size is volume_size+2*pad_size
%   pad_size: integer
%   visu: bool, whether to visualize
%
% dependency:
%   obj_loader, polygon2voxel, plot3D
%
% Author: Charles R. Qi
% Date: Sep 25, 2016

if nargin < 4
    visu = 0;
end

FV = obj_loader(obj_filename);
volume = polygon2voxel(FV, [volume_size, volume_size, volume_size], 'auto', 1, 0);
volume = padarray(volume, [pad_size, pad_size, pad_size]);
volume = int8(volume);
% save(output_filename, 'volume');

if visu
    figure, plot3D(volume);
end

end

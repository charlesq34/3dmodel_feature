function obj2vox_batch(obj_filelist, volume_size, pad_size, output_dir)
% OBJ2VOX_BATCH, convert a list of .obj models to volumes and save them to .mat files
%   obj_filelist: string, path for a txt file each line is 
%       the full path for an OBJ model
%   volume_size: integer, final volume size is volume_size+2*pad_size
%   pad_size: integer
%   output_dir: string, output converted volumes to this folder, 
%       if not exists, we will create it. The name for the .mat files are 
%       .obj names with obj replaced with mat.
%
% Author: Charles R. Qi
% Date: Oct 5, 2016

if nargin < 4
    output_dir = '.';
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

obj_filenames = importdata(obj_filelist);
for k = 1:length(obj_filenames)
    obj_filename = obj_filenames{k};
    instance = obj2vox(obj_filename, volume_size, pad_size);
    [~, filename, ~] = fileparts(obj_filename);
    volume_filename = [filename '.mat'];
    save(fullfile(output_dir, volume_filename), 'instance');
end


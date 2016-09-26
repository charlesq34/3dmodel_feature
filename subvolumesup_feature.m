function feat = subvolumesup_feature(input, params, predict_class)
% SubvolumeSup 3D CNN inference in MATLAB
% feat = SUBVOLUMESUP_FEATURE(input, params, predict_class)
%
%   input: 4-D tensor of size 1x30x30x30 and value 0,1
%       up-direction is positive-z
%   params: struct containing CNN weights and biases
%   predict_class: logical, display predicted class if true
%       in default it's set on.
%
%   feat: extracted feature in dimension 128 x 1
%
% Author: Charles R. Qi
% Date: April 14, 2016


if nargin < 2
    load('subvolumesup_params_ours.mat')
    params = subvolumesup_params;
end
if nargin < 3
    predict_class = true;
end

% conv1
output = conv3d(input, params.conv1w, params.conv1b, 2, 0);
output = output .* (output>0);
% cccp11
output = conv3d(output, params.cccp11w, params.cccp11b, 1, 0);
output = output .* (output>0);
% cccp12
output = conv3d(output, params.cccp12w, params.cccp12b, 1, 0);
output = output .* (output>0);


% conv2
output = conv3d(output, params.conv2w, params.conv2b, 2, 0);
output = output .* (output>0);
% cccp21
output = conv3d(output, params.cccp21w, params.cccp21b, 1, 0);
output = output .* (output>0);
% cccp22
output = conv3d(output, params.cccp22w, params.cccp22b, 1, 0);
output = output .* (output>0);


% conv3
output = conv3d(output, params.conv3w, params.conv3b, 1, 0);
output = output .* (output>0);
% cccp31
output = conv3d(output, params.cccp31w, params.cccp31b, 1, 0);
output = output .* (output>0);
% cccp32
output = conv3d(output, params.cccp32w, params.cccp32b, 1, 0);
output = output .* (output>0);


% fc4
% Since python/C++ flatten is in row-wise order
% need to permute and then reshape here
output = reshape(permute(output,[4 3 2 1]),[],1);
output = params.fc4w * output(:) + reshape(params.fc4b,[],1);
output = output .* (output>0);


% fc5
output = params.fc5w * output(:) + reshape(params.fc5b,[],1);
output = output .* (output>0);
feat = output;

% fc6
output = params.fc6w * output(:) + reshape(params.fc6b,[],1);
[~, predict] = max(output);


if predict_class
    shape_names = {'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',...
        'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp',...
        'laptop','mantel','monitor' 'night_stand','person','piano','plant','radio','range_hood','sink',...
        'sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox'};
    fprintf('Prediction: %s, Confidence: %.6f\n', shape_names{predict}, exp(output(predict))/sum(exp(output)));
end
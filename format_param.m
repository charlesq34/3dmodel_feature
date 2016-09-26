
load('subvolumesup_param.mat');

subvolumesup_params.conv1w = conv1w;
subvolumesup_params.conv1b = conv1b;
subvolumesup_params.cccp11w = cccp11w;
subvolumesup_params.cccp11b = cccp11b;
subvolumesup_params.cccp12w = cccp12w;
subvolumesup_params.cccp12b = cccp12b;

subvolumesup_params.conv2w = conv2w;
subvolumesup_params.conv2b = conv2b;
subvolumesup_params.cccp21w = cccp21w;
subvolumesup_params.cccp21b = cccp21b;
subvolumesup_params.cccp22w = cccp22w;
subvolumesup_params.cccp22b = cccp22b;




subvolumesup_params.conv3w = conv3w;
subvolumesup_params.conv3b = conv3b;
subvolumesup_params.cccp31w = cccp31w;
subvolumesup_params.cccp31b = cccp31b;
subvolumesup_params.cccp32w = cccp32w;
subvolumesup_params.cccp32b = cccp32b;

subvolumesup_params.fc4w = fc4w;
subvolumesup_params.fc4b = fc4b;

subvolumesup_params.fc5w = fc5w;
subvolumesup_params.fc5b = fc5b;

subvolumesup_params.fc6w = fc6w;
subvolumesup_params.fc6b = fc6b;

save('subvolumesup_params_ours.mat', 'subvolumesup_params');

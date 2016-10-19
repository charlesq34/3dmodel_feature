require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
require 'torch'
require 'hdf5'

opt = lapp[[
    --model     (default "logs/model.net")      torch model file path
]]


print('Loading model...')
model = torch.load(opt.model):cuda()
print(model)


-- load h5 file data into memory
function loadDataFile(file_name)
    local current_file = hdf5.open(file_name,'r')
    local current_data = current_file:read('data'):all():float()
    local current_label = torch.squeeze(current_file:read('label'):all():add(1))
    current_file:close()
    return current_data, current_label
end



print('Loading data...')
test_file = 'io/volume_data0.h5'
current_data, current_label = loadDataFile(test_file)
print(#current_data)


print('Starting to test bagging...')
for t = 1,current_data:size(1) do
    --xlua.progress(t, current_data:size(1))
    local inputs = current_data[t]:reshape(1,1,30,30,30)
    local target = current_label[t]
    local outputs = model:forward(inputs:cuda())
    val, idx = torch.max(outputs:double(), 1)
    print(idx)
end

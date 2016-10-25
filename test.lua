require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
require 'torch'
require 'hdf5'

opt_string = [[
    --model         (default "logs/model.net")              torch model file path
    --h5_path       (default "data/volume_data0.h5")        h5 data path
    --gpu_index     (default 0)                             GPU index
    --output_file   (default "output.txt")                  Ouput filename
]]
opt = lapp(opt_string)

-- print help or chosen options
if opt.help == true then
    print('Usage: th train.lua')
    print('Options:')
    print(opt_string)
    os.exit()
else
    print(opt)
end



-- set gpu
cutorch.setDevice(opt.gpu_index+1)

-- output file
outfile = assert(io.open(opt.output_file, "w"))

-- specify which layer's output we would use as feature 
OUTPUT_LAYER_INDEX = 33

print('Loading model...')
model = torch.load(opt.model):cuda()
model:evaluate()
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
test_file = opt.h5_path
current_data, current_label = loadDataFile(test_file)
print(#current_data)


print('Starting testing...')
for t = 1,current_data:size(1) do
    --xlua.progress(t, current_data:size(1))
    local inputs = current_data[t]:reshape(1,1,32,32,32)
    local target = current_label[t]
    local outputs = model:forward(inputs:cuda())
    val, idx = torch.max(outputs:double(), 1)
    --print(idx)
    --print(outputs)
    feat = model:get(OUTPUT_LAYER_INDEX).output:double()
    splitter = ','
    for i=1,feat:size(1) do
        outfile:write(string.format("%.6f", feat[i]))
        if i < feat:size(1) then
            outfile:write(splitter)
        end
    end
    outfile:write('\n')
end

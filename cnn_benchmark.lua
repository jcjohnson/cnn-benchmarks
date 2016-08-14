require 'torch'
require 'nn'

local utils = require 'utils'


local cmd = torch.CmdLine()
-- Model options
cmd:option('-model_t7', 'models/vgg16.t7')
cmd:option('-image_height', 224)
cmd:option('-image_width', 224)
cmd:option('-batch_size', 16)

-- Benchmark options
cmd:option('-num_passes', 10)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)

-- Output options
cmd:option('-output_json', 'outputs/cnn_out.json')

local opt = cmd:parse(arg)
local dtype, use_cudnn, gpu_name, cudnn_version = utils.setup_gpu(opt)
print('Loading model from ' .. opt.model_t7)
local model = torch.load(opt.model_t7)
utils.restore_gradients(model)
model:training()
model:type(dtype)
if use_cudnn then
  cudnn.convert(model, cudnn)
end


local forward_times = {}
local backward_times = {}
local N, C = opt.batch_size, 3
local H, W = opt.image_height, opt.image_width
for t = 1, opt.num_passes + 1 do
  local msg = 'Running iteration %d / %d'
  print(string.format(msg, t - 1, opt.num_passes))
  
  local x = torch.randn(N, C, H, W):type(dtype)
  utils.sync()
  local forward_time = utils.timeit(function() model:forward(x) end)
  if t > 1 then
    -- The first pass does not count since it will allocate
    -- a bunch of memory
    table.insert(forward_times, forward_time)
  end
 
  local dout = torch.randn(#model.output):type(dtype)
  utils.sync()
  local backward_time = utils.timeit(function() model:backward(x, dout) end)
  if t > 1 then
    table.insert(backward_times, backward_time)
  end
end

forward_times = torch.DoubleTensor(forward_times)
backward_times = torch.DoubleTensor(backward_times)
local total_times = forward_times + backward_times

local msg = '%f += %f'
print('Forward:')
print(string.format(msg, forward_times:mean(), forward_times:std()))
print('Backward:')
print(string.format(msg, backward_times:mean(), backward_times:std()))
print('Total:')
print(string.format(msg, total_times:mean(), backward_times:std()))

local json_data = {
  opt = opt,
  forward_times = forward_times:totable(),
  backward_times = backward_times:totable(),
  total_times = total_times:totable(),
  gpu_name = gpu_name,
  cudnn_version = cudnn_version,
}
utils.write_json(opt.output_json, json_data)

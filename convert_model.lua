require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'loadcaffe'
require 'cudnn'
local utils = require 'utils'

local cmd = torch.CmdLine()
cmd:option('-input_prototxt', '')
cmd:option('-input_caffemodel', '')
cmd:option('-input_t7', '')
cmd:option('-output_t7', '')
cmd:option('-backend', 'nn')
cmd:option('-clear_gradients', 1)
local opt = cmd:parse(arg)


if (opt.input_prototxt == '') == (opt.input_t7 == '') then
  error('Must provide one of -input_prototxt or -input_t7')
end

local model = nil
if opt.input_prototxt ~= '' then
  if opt.input_caffemodel == '' then
    error('Must provide both -input_prototxt and -input_caffemodel')
  end
  model = loadcaffe.load(opt.input_prototxt, opt.input_caffemodel, opt.backend)
elseif opt.input_t7 then
  model = torch.load(opt.input_t7)
end
local backend_map = {nn=nn, cudnn=cudnn}
cudnn.convert(model, backend_map[opt.backend])
model:float()
model:clearState()
if opt.clear_gradients == 1 then
  utils.clear_gradients(model)
end
torch.save(opt.output_t7, model)

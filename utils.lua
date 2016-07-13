local cjson = require 'cjson'

local M = {}


function M.setup_gpu(opt)
  local dtype = 'torch.FloatTensor'
  local use_cudnn = false
  local gpu_name = 'cpu'
  local cudnn_version = 'none'
  if opt.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu + 1)
    gpu_name = cutorch.getDeviceProperties(opt.gpu + 1).name
    local msg = 'Running on GPU %d (%s)'
    print(string.format(msg, opt.gpu, gpu_name))
    dtype = 'torch.CudaTensor'
    if opt.use_cudnn == 1 then
      require 'cudnn'
      use_cudnn = true
      cudnn.benchmark = true
      cudnn_version = cudnn.version
      print('Using cuDNN version ' .. tostring(cudnn.version))
    end
  else
    print('Running on CPU')
  end
  return dtype, use_cudnn, gpu_name, cudnn_version
end


function M.sync()
  if cutorch then cutorch.synchronize() end
end


function M.timeit(f)
  M.sync()
  local timer = torch.Timer()
  f()
  M.sync()
  return timer:time().real
end


function M.clear_gradients(m)
  if torch.isTypeOf(m, nn.Container) then
    m:applyToModules(M.clear_gradients)
  end
  if m.weight and m.gradWeight then
    m.gradWeight = m.gradWeight.new()
  end
  if m.bias and m.gradBias then
    m.gradBias = m.gradBias.new()
  end
end


function M.restore_gradients(m)
  if torch.isTypeOf(m, nn.Container) then
    m:applyToModules(M.restore_gradients)
  end
  if m.weight and m.gradWeight then
    m.gradWeight = m.gradWeight.new(#m.weight):zero()
  end
  if m.bias and m.gradBias then
    m.gradBias = m.gradBias.new(#m.bias):zero()
  end
end


function M.read_json(path)
  local file = assert(io.open(path, 'r'))
  local text = file:read()
  local info = cjson.decode(file:read())
  file:read()
  return info
end


function M.write_json(path, data)
  local file = assert(io.open(path, 'w'))
  file:write(cjson.encode(data))
  file:close()
end


return M

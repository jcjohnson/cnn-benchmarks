import argparse
import time
import torch
import torch.nn as nn
import torchvision.models
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--models', default='alexnet')
parser.add_argument('--num_warmup', default=3, type=int)
parser.add_argument('--num_batches', default=10, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--mode', default='train', choices=['train', 'test'])

parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--cudnn_benchmark', default=1, type=int)
parser.add_argument('--precisions', default='fp16,fp32')
parser.add_argument('--batchnorm_workaround', default=1, type=int)


def get_dtype(use_gpu, precision):
  dtypes = {
    (1, 'fp16'): torch.cuda.HalfTensor,
    (1, 'fp32'): torch.cuda.FloatTensor,
    (1, 'fp64'): torch.cuda.DoubleTensor,
    (0, 'fp16'): torch.HalfTensor,
    (0, 'fp32'): torch.FloatTensor,
    (0, 'fp64'): torch.DoubleTensor,
  }
  key = (use_gpu, precision)
  return dtypes[key]


class DummyBatchNorm2d(nn.Module):
  def __init__(self, channels, *args, **kwargs):
    super(DummyBatchNorm2d, self).__init__()
    self.weight = nn.Parameter(torch.zeros(channels))
    self.bias = nn.Parameter(torch.ones(channels))
  def forward(self, x):
    return x



def main(args):
  if args.batchnorm_workaround == 1:
    torch.nn.BatchNorm2d = DummyBatchNorm2d

  if args.use_gpu == 1:
    if args.cudnn_benchmark == 1:
      torch.backends.cudnn.benchmark = True
    print('Using cuDNN version ', torch.backends.cudnn.version())
  
  for model in args.models.split(','):
    for precision in args.precisions.split(','):
      dtype = get_dtype(args.use_gpu, precision)
      print('Running %s with dtype %s' % (model, dtype))
      forward_times, backward_times = run_model(args, model, dtype)
      total_times = forward_times + backward_times
      f_mean, f_std = forward_times.mean(), forward_times.std()
      b_mean, b_std = backward_times.mean(), backward_times.std()
      t_mean, t_std = total_times.mean(), total_times.std()
      print('Forward: %.2f ms +- %.2f ms' % (f_mean, f_std))
      print('Backward: %.2f ms +- %.2f ms' % (b_mean, b_std))
      print('Total: %.2f ms +- %.2f ms' % (t_mean, t_std))


def run_model(args, model, dtype):
  if not hasattr(torchvision.models, model):
    raise ValueError('Invalid model "%s"' % model)
  model = getattr(torchvision.models, model)().type(dtype)
  if args.mode == 'train':
    model.train()
  elif args.mode == 'test':
    model.eval()

  N, C, H, W = args.batch_size, 3, args.image_size, args.image_size
  forward_times, backward_times = [], []
  total_batches = args.num_warmup + args.num_batches
  for t in range(total_batches):
    print('Running batch %d / %d' % (t + 1, total_batches))
    x = torch.randn(N, C, H, W)
    x = Variable(x.type(dtype))

    torch.cuda.synchronize()
    t0 = time.time()
    y = model(x)
    torch.cuda.synchronize()
    t1 = time.time()

    dy = torch.randn(y.size()).type(dtype)

    torch.cuda.synchronize()
    t2 = time.time()
    dx = y.backward(dy)
    torch.cuda.synchronize()
    t3 = time.time()

    if t >= args.num_warmup:
      forward_times.append(t1 - t0)
      backward_times.append(t3 - t2)

  forward_times = torch.Tensor(forward_times).mul_(1000)
  backward_times = torch.Tensor(backward_times).mul_(1000)
  return forward_times, backward_times


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


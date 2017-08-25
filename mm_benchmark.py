import argparse
import time
import torch


parser = argparse.ArgumentParser()


def main(args):
  sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
  dtypes = [torch.cuda.FloatTensor, torch.cuda.HalfTensor]
  for dtype in dtypes:
    print('Running with dtype %s' % dtype)
    for size in sizes:
      times = mm_benchmark(size, dtype)
      mean, std = times.mean(), times.std()
      print('Size %d took %.2f ms (pm %2.f ms)' % (size, mean, std))


def mm_benchmark(size, dtype, num_runs=10, num_warmup=3):
  a = torch.randn(size, size).type(dtype)
  b = torch.randn(size, size).type(dtype)
  c = torch.randn(size, size).type(dtype)
  times = []
  for t in range(num_runs + num_warmup):
    torch.cuda.synchronize()
    t0 = time.time()
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    t1 = time.time()
    if t >= num_warmup:
      times.append(1000 * (t1 - t0))
  return torch.Tensor(times)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


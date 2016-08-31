import json, os, argparse, itertools, math
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='outputs')
parser.add_argument('--include_std', default=0)
args = parser.parse_args()


# Maps the cuDNN version reported by torch.cudnn to a more friendly string
cudnn_map = {
  5005: '5.0.05',
  5105: '5.1.05',
  4007: '4.0.07',
  'none': 'None',
}

# Maps the GPU name reported by the driver to a more friendly string
gpu_name_map = {
  'Tesla P100-SXM2-16GB': 'Tesla P100',
  'TITAN X': 'Pascal Titan X',
  'GeForce GTX TITAN X': 'Maxwell Titan X',
  'GeForce GTX 1080': 'GTX 1080',
  'cpu': 'CPU: Dual Xeon E5-2630 v3',
}


def main(args):
  # Load all the results
  results = []
  for dirpath, dirnames, fns in os.walk(args.results_dir):
    for fn in fns:
      if not fn.endswith('.json'): continue
      with open(os.path.join(dirpath, fn), 'r') as f:
        results.append(json.load(f))

  all_values = defaultdict(set)
  keyed_results = {}
  
  for result in results:
    gpu_name = result['gpu_name']
    cudnn_version = result['cudnn_version']
    model = result['opt']['model_t7']
    
    batch_size = result['opt']['batch_size']
    im_width = result['opt']['image_width']
    im_height = result['opt']['image_height']
    input_size = '%d x 3 x %d x %d' % (batch_size, im_height, im_width)
    
    model = os.path.splitext(os.path.basename(model))[0]
    keyed_results[(gpu_name, cudnn_version, model)] = result
    
    all_values['gpu_name'].add(gpu_name)
    all_values['cudnn_version'].add(cudnn_version)
    all_values['model'].add(model)
    all_values['input_size'].add(input_size)
  
  for k, vs in all_values.iteritems():
    print k
    for v in vs:
      print '  %s' % v
  
  markdown_tables = {}
  
  for model in all_values['model']:
    for input_size in all_values['input_size']:
      table_header = '|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|'
      table_header2 = '|---|---|---:|---:|---:|'
      table_lines = {}
      for gpu_name in all_values['gpu_name']:
        for cudnn_version in all_values['cudnn_version']:
          k = (gpu_name, cudnn_version, model)
          if k not in keyed_results: continue
          result = keyed_results[k]

          cudnn_str = cudnn_map[cudnn_version]
          cudnn_str = cudnn_map.get(cudnn_version, cudnn_version)
          gpu_str = gpu_name_map.get(gpu_name, gpu_name)

          f_mean = mean(result['forward_times']) * 1000
          f_std = std(result['forward_times']) * 1000
          b_mean = mean(result['backward_times']) * 1000
          b_std = std(result['backward_times']) * 1000
          t_mean = mean(result['total_times']) * 1000
          t_std = std(result['total_times']) * 1000

          if args.include_std == 1:
            f_str = '%.2f += %.2f' % (f_mean, f_std)
            b_str = '%.2f += %.2f' % (b_mean, b_std)
            t_str = '%.2f += %.2f' % (t_mean, t_std)
          else:
            f_str = '%.2f' % f_mean
            b_str = '%.2f' % b_mean
            t_str = '%.2f' % t_mean
          table_lines[t_mean] = '|%s|%s|%s|%s|%s|' % (
                gpu_str, cudnn_str, f_str, b_str, t_str)

      table_lines = [table_lines[k] for k in sorted(table_lines)]
      table_lines = [table_header, table_header2] + table_lines
      model_batch_str = '%s (input %s)' % (model, input_size)
      markdown_tables[model_batch_str] = table_lines

  for model, table_lines in markdown_tables.iteritems():
    print model
    for line in table_lines:
      print line
    print


def mean(xs):
  return float(sum(xs)) / len(xs)


def std(xs):
  m = mean(xs)
  diffs = [x - m for x in xs]
  var = sum(d ** 2.0 for d in diffs) / (len(xs) - 1)
  return math.sqrt(var)
  
        
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


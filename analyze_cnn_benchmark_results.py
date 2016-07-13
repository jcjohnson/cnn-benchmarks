import json, os, argparse, itertools, math
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='outputs/cnn')
parser.add_argument('--include_std', default=0)
parser.add_argument('--output_markdown_tables', default='tables.md')
args = parser.parse_args()


def main(args):
  # Load all the results
  results = []
  for fn in os.listdir(args.results_dir):
    if not fn.endswith('.json'): continue
    with open(os.path.join(args.results_dir, fn), 'r') as f:
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
      table_header = '|GPU|Forward (ms)|Backward (ms)|Total (ms)|'
      table_header2 = '|---|---|---|---|'
      table_lines = {}
      for gpu_name in all_values['gpu_name']:
        for cudnn_version in all_values['cudnn_version']:
          k = (gpu_name, cudnn_version, model)
          if k not in keyed_results: continue
          result = keyed_results[k]

          cudnn_str = '(cuDNN %s)' % cudnn_version
          if cudnn_version == 'none':
            cudnn_str = '(nn)'
          gpu_str = '%s %s' % (gpu_name, cudnn_str)

          f_mean = mean(result['forward_times']) * 1000
          f_std = std(result['forward_times']) * 1000
          b_mean = mean(result['forward_times']) * 1000
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
          table_lines[t_mean] = '|%s|%s|%s|%s|' % (gpu_str, f_str, b_str, t_str)

      table_lines = [table_lines[k] for k in sorted(table_lines)]
      table_lines = [table_header, table_header2] + table_lines
      model_batch_str = '%s (input %s)' % (model, input_size)
      markdown_tables[model_batch_str] = table_lines

  with open(args.output_markdown_tables, 'w') as f:
    for model, table_lines in markdown_tables.iteritems():
      f.write('%s\n'% model)
      for line in table_lines:
        f.write('%s\n'% line)
      f.write('\n')

def mean(xs):
  return float(sum(xs)) / len(xs)


def std(xs):
  m = mean(xs)
  diffs = [x - m for x in xs]
  var = sum(d ** 2.0 for d in diffs) / (len(xs) - 1)
  return math.sqrt(var)
  

def grouped_bar(labels, vals, group_names, stds=None,
                width=0.8, big_spacing=1.0, small_spacing=0.0):
  pass
        
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

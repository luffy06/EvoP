import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    filename = os.path.basename(args.log_file)

    with open(args.log_file, 'r') as f:
        lines = f.readlines()

    nrm_blocks = []
    loss = []
    nrm_block_orders = []
    ppl = {}
    acc = {}
    for line in lines:
        if 'Number of removed blocks' in line:
            nrm_blocks.append(line.split(':')[-1].strip())
        elif 'Best loss' in line:
            loss.append(line.split(':')[-1].strip().strip('[').strip(']'))
        elif 'Best skip layers' in line:
            line = line.split(':')[-1].strip().strip('[').strip(']')
            line = line.split(',')
            line = list(filter(lambda x: x.strip() != '', line))
            orders = [x for x in line]
            nrm_block_orders.append(orders)
        elif 'PPL = ' in line:
            line = line.split(':')[-1].strip().split()
            dataset = line[0]
            if dataset not in ppl:
                ppl[dataset] = []
            ppl[dataset].append(line[-1])
        elif 'acc,none' in line:
            line = ' '.join(line.split(' ')[1:]).strip()
            line = line.replace('\'', '\"')
            results = json.loads(line.strip())
            task = results['alias']
            res = results['acc,none']
            if task not in acc:
                acc[task] = []
            acc[task].append(str(res))

    num = len(nrm_blocks)
    assert num == len(loss), f"There are {len(loss)} loss results, which is not equal to {num}"
    assert num == len(nrm_block_orders), f"There are {len(nrm_block_orders)} nrm_block_orders, which is not equal to {num}"
    for dataset in ppl:
        assert num == len(ppl[dataset])
    for task in acc:
        assert num == len(acc[task])

    with open(os.path.join(args.output_dir, 'result_' + filename.split('.')[0] + '.txt'), 'w') as f:
        f.write('\t'.join(nrm_blocks) + '\n')
        f.write('\t'.join(loss) + '\n')
        for dataset in ppl:
            f.write('\t'.join(ppl[dataset]) + '\n')
        for task in acc:
            f.write(task + '\n')
            f.write('\t'.join(acc[task]) + '\n')
        for orders in nrm_block_orders:
            f.write('\t'.join(orders) + '\n')
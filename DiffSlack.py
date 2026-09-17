import os
import yaml
import torch
import time
from models.DiffSlack_Trainer import DiffSlack_Trainer
from DataLoader.dataload import My_Dataset
from DataLoader.dataload_IL import My_Dataset_IL, openjson
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
import json
import gc
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--initialization',
        choices=('learned', 'zero', 'constant', 'analytic'),
    )
    parser.add_argument('--constant', type=float)
    parser.add_argument('--load-dir', default='save_dir/DiffSlack/epoch_299.pth')
    parser.add_argument('--data-dir', default='/home/qian/dataset_V7/')
    parser.add_argument('--run-name')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument(
        '--profile-timing', action='store_true',
        help='Run a second instrumented projection pass for timing breakdown.',
    )
    parser.add_argument(
        '--failure-diagnosis', action='store_true',
        help='Diagnose exact-collision failures after the normal test finishes.',
    )
    parser.add_argument('--diag-i-max', type=int)
    parser.add_argument(
        '--disable-sample-overfit', action='store_true',
        help='Skip the sample-specific overfit stage during failure diagnosis.',
    )
    parser.add_argument(
        '--failure-sample-ids',
        help=(
            'Load a previous failure_sample_ids.pt and rerun diagnosis only '
            'for those samples.'
        ),
    )
    args = parser.parse_args()

    config_path = 'configs/DiffSlack.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if args.initialization is not None:
        config['slack_initialization'] = args.initialization
    if args.constant is not None:
        config['slack_initialization_constant'] = args.constant
    config['profile_timing'] = args.profile_timing
    if args.failure_diagnosis:
        config['enable_failure_diagnosis'] = True
    if args.diag_i_max is not None:
        config['diag_i_max'] = args.diag_i_max
    if args.disable_sample_overfit:
        config['enable_sample_overfit'] = False
    if args.failure_sample_ids is not None:
        config['enable_failure_diagnosis'] = True
        
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dataset = My_Dataset(data_dir=args.data_dir, length=200000)
    
    train_size = int(len(dataset) * 0.6)
    val_size = int(len(dataset) * 0.3)
    test_size = int(len(dataset) * 0.1)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
    test_id = 0
    if args.failure_sample_ids is not None:
        failure_id_info = torch.load(
            args.failure_sample_ids, map_location='cpu', weights_only=False
        )
        if isinstance(failure_id_info, dict):
            failure_sample_ids = failure_id_info['sample_ids']
            config['failure_diagnosis_total_test_samples'] = int(
                failure_id_info.get('total_test_samples', len(failure_sample_ids))
            )
        else:
            failure_sample_ids = failure_id_info
            config['failure_diagnosis_total_test_samples'] = len(
                failure_sample_ids
            )
        if isinstance(failure_sample_ids, torch.Tensor):
            failure_sample_ids = failure_sample_ids.tolist()
        failure_sample_ids = [int(sample_id) for sample_id in failure_sample_ids]
        if not failure_sample_ids:
            raise ValueError(
                f'No failure sample IDs found in {args.failure_sample_ids}'
            )
        invalid_ids = [
            sample_id for sample_id in failure_sample_ids
            if sample_id < 0 or sample_id >= len(dataset)
        ]
        if invalid_ids:
            raise ValueError(
                f'Failure sample IDs outside dataset range: {invalid_ids}'
            )
        test_dataset = torch.utils.data.Subset(dataset, failure_sample_ids)
        print(
            f'Loaded {len(failure_sample_ids)} failure sample IDs from '
            f'{args.failure_sample_ids}'
        )
    else:
        test_dataset = torch.utils.data.Subset(
            dataset, range(test_id, test_id + test_size)
        )
    
    val_dataset = torch.utils.data.Subset(dataset, range(test_size, test_size + val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - train_size, len(dataset)))

    base_name = args.run_name or f"DiffSlack_{config['slack_initialization']}"
    
    # id = time.strftime("%Y%m%d-%H%M%S")
    id = config['seed']
    
    log_dir = f'logs/{base_name}/{base_name}_{id}'
    save_dir = f'save_dir/{base_name}/{base_name}_{id}'
    save_config_file = f'logs/{base_name}/{base_name}_{id}/config.yaml'
    save_prob_file = os.path.join(log_dir, 'prob.py')
    
    os.makedirs(os.path.dirname(save_config_file), exist_ok=True)
    if not os.path.exists(save_config_file):
        with open(save_config_file, 'w') as f:
            yaml.dump(config, f)
    import shutil
    if not os.path.exists(save_prob_file):
        shutil.copy('utils/prob.py', save_prob_file)
    
    test_config = config.copy()
    trainer = DiffSlack_Trainer(config=test_config,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset,
                        save_dir=save_dir,
                        load_dir=args.load_dir,
                        log_dir=log_dir,
                    )
    if args.failure_sample_ids is not None:
        trainer.test(
            test_hard=True,
            result_name='test_results_failure_subset.txt',
        )
    elif args.test_only:
        trainer.test(test_hard=True)
    else:
        trainer.train(begin_epoch=config['begin_epoch'])
    # trainer.save_path_data('./carla/paths/DiffSlack')
    # trainer.test_visualization(os.path.join(log_dir, 'test_visualization'))

if __name__ == "__main__":
    main()

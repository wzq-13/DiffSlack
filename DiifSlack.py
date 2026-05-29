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
def main():
    config_path = 'configs/APF_hard.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dataset = My_Dataset(data_dir='./dataset/', length=200000)
    
    train_size = int(len(dataset) * 0.6)
    val_size = int(len(dataset) * 0.3)
    test_size = 200 #int(len(dataset) * 0.1)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
    test_id = 0
    test_dataset = torch.utils.data.Subset(dataset, range(test_id, test_id + test_size))
    
    val_dataset = torch.utils.data.Subset(dataset, range(test_size, test_size + val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - train_size, len(dataset)))

    base_name = 'APF_hard'
    
    # id = time.strftime("%Y%m%d-%H%M%S")
    id = '20260522-143313'
    # id = config['seed']
    
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
                        load_dir='save_dir/A-APF_hard/A-APF_hard_1/epoch_399.pth',
                        log_dir=log_dir,
                    )
    # trainer.train(begin_epoch=300)
    trainer.test(test_hard=True)
    # trainer.save_path_data('./carla/paths/DiffSlack')
    # trainer.test_visualization(os.path.join(log_dir, 'test_visualization'))

if __name__ == "__main__":
    main()

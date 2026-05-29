import os
import yaml
import torch
import time
from models.DC3_Trainer import DC3_Trainer
from DataLoader.dataload import My_Dataset
from DataLoader.dataload_IL import My_Dataset_IL, openjson
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
def main():
    config_path = 'configs/DC3.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset = My_Dataset(data_dir='./dataset/', length=200000)
    
    train_size = int(len(dataset) * 0.6)
    val_size = int(len(dataset) * 0.3)
    test_size = 1 #int(len(dataset) * 0.1)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
    test_dataset = torch.utils.data.Subset(dataset, range(74, 74+test_size))
    
    val_dataset = torch.utils.data.Subset(dataset, range(test_size, test_size + val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - train_size, len(dataset)))

    base_name = 'DC3'
    
    time_now = time.strftime("%Y%m%d-%H%M%S")
    time_now = '20260512-194846'
    
    log_dir = f'logs/{base_name}/{base_name}_{time_now}'
    save_dir = f'save_dir/{base_name}/{base_name}_{time_now}'
    save_config_file = f'logs/{base_name}/{base_name}_{time_now}/config.yaml'
    save_prob_file = os.path.join(log_dir, 'prob.py')
    
    os.makedirs(os.path.dirname(save_config_file), exist_ok=True)
    if not os.path.exists(save_config_file):
        with open(save_config_file, 'w') as f:
            yaml.dump(config, f)
    import shutil
    if not os.path.exists(save_prob_file):
        shutil.copy('utils/prob.py', save_prob_file)
    
    trainer = DC3_Trainer(config=config,
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      save_dir=save_dir,
                      load_dir='save_dir/A-DC3/DC3_20260512-194846/epoch_399.pth',
                      log_dir=log_dir,
                    )
    # trainer.train(begin_epoch=config['begin_epoch'])
    trainer.test()
    # trainer.test_visualization(os.path.join(log_dir, 'test_visualization'))
    # trainer.save_path_data('./carla/paths/DC3')

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()
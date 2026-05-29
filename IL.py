import os
import yaml
import torch
import time
import argparse
from models.IL_Trainer import IL_Trainer
from DataLoader.dataload import My_Dataset
from DataLoader.dataload_IL import My_Dataset_IL, openjson
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
import json
def main():
    config_path = 'configs/IL.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    batch_size = config['batch_size']
    data_dir = './dataset_NMPC-label/'
    filelist = './IL.json'
    datafilelist = openjson(filelist)['files']
    # dataset = My_Dataset2(data_dir=data_dir, input_dir=input_dir)

    dataset = My_Dataset_IL(data_dir=data_dir, datafilelist=datafilelist)
    dataset_all = My_Dataset(data_dir='./dataset/', length=5000)
    
    train_size = int(len(dataset) * 0.8)
    val_size = int((len(dataset) * 0.2))
    test_size = 1
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
    test_dataset = torch.utils.data.Subset(dataset_all, range(74, 74+test_size))
    
    val_dataset = torch.utils.data.Subset(dataset, range(0, val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(val_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=8,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=8,persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True,num_workers=8,persistent_workers=False)

    if config['use_soft_constraints']:
        base_name = 'IL_soft'
    else:
        base_name = 'IL_pure'
    
    time_now = time.strftime("%Y%m%d-%H%M%S")
    time_now = '20260508-095641'
    
    log_dir = f'logs/{base_name}/{base_name}_{time_now}'
    save_dir = f'save_dir/{base_name}/{base_name}_{time_now}'
    save_config_file = f'logs/{base_name}/{base_name}_{time_now}/config.yaml'
    save_prob_file = os.path.join(log_dir, 'prob.py')
    
    os.makedirs(os.path.dirname(save_config_file), exist_ok=True)
    if not os.path.exists(save_config_file):
        with open(save_config_file, 'w') as f:
            yaml.dump(config, f)
    else:
        print(f"Config file {save_config_file} already exists. Skipping saving config.")
    import shutil
    if not os.path.exists(save_prob_file):
        shutil.copy('utils/prob.py', save_prob_file)
    else:
        print(f"Prob file {save_prob_file} already exists. Skipping copying prob.py.")

    trainer = IL_Trainer(config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      save_dir=save_dir,
                      load_dir='save_dir/IL_soft/IL_soft_20260508-095641/epoch_399.pth',
                      log_dir=log_dir,
                    )
    # trainer.test(test_loader)
    trainer.test_visualization(os.path.join(log_dir, 'test_visualization'))
    # trainer.train()
    # trainer.save_path_data(''./carla/paths/IL)
    
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()
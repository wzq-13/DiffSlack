import os
import yaml
import torch
import time
from models.IL_Hard_Trainer import IL_Hard_Trainer
from DataLoader.dataload import My_Dataset
from DataLoader.dataload_IL import My_Dataset_IL, openjson
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
import json
def main():
    config_path = 'configs/IL hard.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    batch_size = config['batch_size']
    data_dir = '/home/qian/dataset_V7_NMPC-label/'
    filelist = './IL.json'
    datafilelist = openjson(filelist)['files']
    # dataset = My_Dataset2(data_dir=data_dir, input_dir=input_dir)
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dataset = My_Dataset_IL(data_dir=data_dir, datafilelist=datafilelist)
    dataset_all = My_Dataset(data_dir='/home/qian/dataset_V7/', length=5000)
    
    train_size = int(len(dataset) * 0.8)
    val_size = int((len(dataset) * 0.2))
    test_size = 100
    test_id = 0
    test_dataset = torch.utils.data.Subset(dataset, range(test_id, test_id + test_size))
    
    val_dataset = torch.utils.data.Subset(dataset, range(0, val_size))
    train_dataset = torch.utils.data.Subset(dataset, range(val_size, len(dataset)))

    base_name = 'A-IL_hard'
    
    time_now = time.strftime("%Y%m%d-%H%M%S")
    time_now = '20260508-111456'
    
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

    trainer = IL_Hard_Trainer(config=config,
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      save_dir=save_dir,
                      load_dir='save_dir/IL_hard/IL_hard_20260508-111456/epoch_399.pth',
                      log_dir=log_dir,
                    )
    # trainer.train(begin_epoch=300)
    # trainer.test()
    trainer.test_visualization(os.path.join(log_dir, 'test_visualization'))

if __name__ == "__main__":
    main()
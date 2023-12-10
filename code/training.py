import torch
import numpy as np
import random
import yaml
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import wandb
import yaml
from model import ConvBERT
from trainer import ConvBERTTrainer
from code.dataset.train_dataset_ztf import ZTFDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate([b for b in batch if b is not None])

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_data_loaders(config, train_ids, valid_ids, sample_weights):
    fineTune_dataset = ZTFDataset(
        config['pathZTF'], 
        config['labels_path'],
        classes=config['classesZTF'], 
        seq_len=config['max_length'], 
        list_IDs=[train_ids[i] for i in range(len(train_ids))], 
        Transform=True
    )

    valid_fineTune_dataset = ZTFDataset(
        config['pathZTF'], 
        config['labels_path'],
        classes=config['classesZTF'], 
        seq_len=config['max_length'], 
        list_IDs=[valid_ids[i] for i in range(len(valid_ids))], 
        Transform=True
    )

    train_subsampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        fineTune_dataset, sampler=train_subsampler,
        batch_size=config['batch_size'], drop_last=False,
        collate_fn=collate_fn, num_workers=12
    )

    valid_loader = DataLoader(
        valid_fineTune_dataset, shuffle=True,
        batch_size=config['batch_size'], drop_last=False,
        collate_fn=collate_fn, num_workers=12
    )

    return train_loader, valid_loader

def training_the_model(config, train_loader, valid_loader, run_name, model_id):
    conv_bert = ConvBERT(hidden=config['hidden_size'], n_layers=config['layers'],
                         attn_heads=config['attn_heads'], dropout=config['dropout'])

    trainer = ConvBERTTrainer(
        conv_bert, 
        conv_bert,  # assuming you are passing the same model for some reason
        config['num_classes'],
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        lr=config['learning_rate'],
        fold=run_name,
        modelId=f'{model_id}-ZTF',
        momentum=0.99, 
        preTrain=False, 
        file_path=config['input_path']
    )

    if config['input_path']:
        trainer.loadModel(config['input_path'])

    max_accuracy = 0
    for epoch in range(config['epochs']):
        _, _, valid_accuracy, _ = trainer.train()
        if max_accuracy < valid_accuracy:
            max_accuracy = valid_accuracy
            trainer.save(epoch, config['finetune_path'])

def main():
    setup_seed(123)
    config = load_config('config.yaml')
    list_data_train_ztf = []  # TODO: Initialize this list properly
    kfold = KFold(n_splits=config['k_folds'], shuffle=True)
    print('--------------------------------')

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(list_data_train_ztf)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        run = wandb.init(project='Project Name')
        print(f'The name of the run is {run.name}')

        fine_weights = [0.4016, 2.2522, 5.1282]
        sample_weights = [fine_weights[data['class_label']] for data in ZTFDataset(...)]  # Initialize this list correctly

        train_loader, valid_loader = create_data_loaders(config, train_ids, valid_ids, sample_weights)

        print("Creating Dataloader...")
        for model_id in range(3):
            print(f"training_the_model model {model_id}")
            training_the_model(config, train_loader, valid_loader, run.name, model_id)

        run.finish()

if __name__ == "__main__":
    main()

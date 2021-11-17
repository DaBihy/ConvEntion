from typing import Counter
import torch
from torch.utils.data import DataLoader
from model import SBERT
from dataset.dataset_wrapper_finetune import DataSetWrapper
from trainer import SBERTFineTuner
from dataset.finetune_dataset_SDSSFewChanSpectro import FinetuneDataset
import numpy as np
import random
import argparse
import itertools
import pandas as pd
from astropy.table import Table
from sklearn.model_selection import KFold

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    # print('This is dic ')
    # print(listOfKeys)  
    return  listOfKeys

    
def get_id_label(meta_path):
    dict_id_label = {}
    all_transients = Table.read(meta_path)
    for transient in all_transients :
        if str(transient['Classification']) == 'Unknown': continue #we exclude Unknown from our dataset 
        non_supernova =['AGN', 'Unknown', 'Variable']
        train_snII = ['pSNII', 'pSNIbc','zSNII','zSNIbc']
        train_snIa = ['pSNIa','zSNIa']
        Valid_snIa = ['SNIa?','SNIa']
        # print(str(transient['Classification']))
        if str(transient['Classification']) in non_supernova:
            dict_id_label[int(transient['CID'])] = str(transient['Classification'])
        elif str(transient['Classification']) in train_snII:
            dict_id_label[int(transient['CID'])] = 'TrainSNII'
        elif str(transient['Classification']) in train_snIa:
            dict_id_label[int(transient['CID'])] = 'TrainSNIa'
        elif str(transient['Classification']) in Valid_snIa:
            dict_id_label[int(transient['CID'])] = 'ValidSNIa'
        else :
            dict_id_label[int(transient['CID'])] = 'ValidSNII'
    # print('This is dic ')
    # print(dict_id_label)        
    return dict_id_label

setup_seed(123)

def Config():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--file_path",
        default='/home/barrage/anass/sdss/v3/stamps_v3/',
        type=str,
        required=False,
        help="The input data path.",
    )

    parser.add_argument(
        "--accur_path",
        default='/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/TrainGroupsFineTuneLossWeight-256-1/CSVStats/stats.csv',
        type=str,
        required=False,
        help="The path for saving the rsults of the metrics",
    )
    parser.add_argument(
        "--labels_path",
        default='/home/barrage/anass/sdss/master_data.fits',
        type=str,
        required=False,
        help="The labels path.",
    )
    parser.add_argument(
        "--pretrain_path",
        default="/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints/TrainGroupsFineTuneLossWeight-256-1/",
        type=str,
        required=False,
        help="The storage path of the pre-trained model.",
    )
    parser.add_argument(
        "--finetune_path",
        default='/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints/TrainGroupsFineTuneLossWeight-256-1/',
        type=str,
        required=False,
        help="The output directory where the fine-tuning checkpoints will be written.",
    )
    parser.add_argument(
        "--max_length",
        default=99,
        type=int,
        help="The maximum length of input time series. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_features",
        default=10,
        type=int,
        help="",
    )
    parser.add_argument(
        "--num_classes",
        default=4,
        type=int,
        help="",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="",
    )
    parser.add_argument(
        "--hidden_size",
        default=[256, 8, 9, 9, 99//3],
        type=int,
        help="",
    )
    parser.add_argument(
        "--layers",
        default=2,
        type=int,
        help="",
    )
    parser.add_argument(
        "--attn_heads",
        default=4,
        type=int,
        help="",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.5,
        type=float,
        help="",
    )
    parser.add_argument(
        "--test_rate",
        default=0.1,
        type=float,
        help="",
    )
    parser.add_argument(
        "--valid_rate",
        default=0.2,
        type=float,
        help="",
    )
    return parser.parse_args()

if __name__ == "__main__":
    config = Config()
    #classes = {'AGN':0, 'SLSN':1, 'SNII':1, 'SNIa':1, 'SNIa?':1, 'SNIb':1, 'SNIc':1, 
    #'Variable':2, 'pSNII':1, 'pSNIa':1, 'pSNIbc':1, 'zSNII':1,
    # 'zSNIa':1, 'zSNIbc':1}
   
    # Generating data loaders 
    Train_path = '/home/barrage/plasticc/plasticc/ConvBERTBandSep3DCNNReguWeigthedFineTune/code/train_data_Oversample.csv'
    FineTune_Path = '/home/barrage/plasticc/plasticc/ConvBERTBandSep3DCNNReguWeigthedFineTune/code/FineTune_data_Oversample.csv'

    list_data_train = pd.read_csv(Train_path)['smaples']
    list_data_FineTune = pd.read_csv(FineTune_Path)['smaples']

    # Configuration options
    k_folds = 5

#   Dataframes to store the stats 
    accuu = {'Model_Name':[], 'OAccuracy':[], 'AAccuracy':[], 'F1Score':[]}
    df_accuracies_All =  pd.DataFrame.from_dict(accuu)
  
    torch.manual_seed(42)

    classes = {'AGN':0,  
     'SNIa':1, 'SNIa?':1, 'zSNIa':1, 'pSNIa':1, 
     'Variable':2, 
     'pSNIbc':3, 'zSNII':3, 'zSNIbc':3 ,'SLSN':3,  'SNIb':3,'pSNII':3, 'SNIc':3,'SNII':3}

    class_weigth_train = [2.10674779, 1.31526243, 0.59054264, 0.93312102]
    class_weigth_fine = [1.5115894,  1.26805556, 0.42465116, 5.12921348]

    print("Loading Data sets...")
    # train_dataset = FinetuneDataset(config.file_path, config.labels_path,
    #  classes=classes, seq_len=config.max_length, list_IDs=list_data_train, Transform=True)

    # fineTune_dataset = FinetuneDataset(config.file_path, config.labels_path,
    #  classes=classes, seq_len=config.max_length, list_IDs=list_data_FineTune, Transform=True)

    print(f'Number of training sampls: {len(list_data_train)} and Number of fineTune samples : {len(list_data_FineTune)} ')

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, ((train_ids, valid_ids), (train_ids_fine, valid_ids_fine)) in enumerate(zip(kfold.split(list_data_train), kfold.split(list_data_FineTune))):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_weights = [11.0619469 ,  6.90607735 , 3.10077519,  4.89955904]
        # train_weights = [7.0619469 ,  6.90607735 , 3.10077519,  4.89955904]
        sample_train_weights= [0]*len(train_ids)
        fine_weights = [11.03752759,  9.25925926,  3.10077519, 37.45318352]
        # fine_weights = [3.03752759,  6.25925926,  3.10077519, 15.45318352]
        # fine_weights =[0.00110375, 0.00092593, 0.00031008, 0.00374532]
        sample_fine_weights= [0]*len(train_ids_fine)

        train_dataset = FinetuneDataset(config.file_path, config.labels_path,
        classes=classes, seq_len=config.max_length, list_IDs=list(list_data_train[train_ids]), Transform=True)

        valid_train_dataset = FinetuneDataset(config.file_path, config.labels_path,
        classes=classes, seq_len=config.max_length, list_IDs=list(list_data_train[valid_ids]), Transform=True)

        fineTune_dataset = FinetuneDataset(config.file_path, config.labels_path,
        classes=classes, seq_len=config.max_length, list_IDs=list(list_data_FineTune[train_ids_fine]), Transform=True)

        valid_fineTune_dataset = FinetuneDataset(config.file_path, config.labels_path,
        classes=classes, seq_len=config.max_length, list_IDs=list(list_data_FineTune[valid_ids_fine]), Transform=True)
        
        # for idx , data in enumerate(train_dataset):
        #     sample_train_weights[idx] = train_weights[data['class_label']]

        for idx , data in enumerate(fineTune_dataset):
            sample_fine_weights[idx] = fine_weights[data['class_label']]

        # Sample elements randomly from a given list of ids, no replacement.
        # train_subsampler = torch.utils.data.WeightedRandomSampler(sample_train_weights, len(sample_train_weights), replacement=True)
#         valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        train_subsampler_fine = torch.utils.data.WeightedRandomSampler(sample_fine_weights, len(sample_fine_weights), replacement=True)
#         valid_subsampler_fine = torch.utils.data.SubsetRandomSampler(valid_ids_fine)
  
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)
        
        print("Creating Dataloader...")
        # train_data_loader = DataLoader(train_dataset,sampler=train_subsampler,
        #                             batch_size=config.batch_size, drop_last=False,
        #                                 collate_fn=collate_fn, num_workers=12 )

        train_data_loader = DataLoader(train_dataset, shuffle=True,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 )

        valid_data_loader = DataLoader(valid_train_dataset, shuffle=True,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 )

        train_data_loader_fine = DataLoader(fineTune_dataset,sampler=train_subsampler_fine,
                                    batch_size=config.batch_size, drop_last=False,
                                        collate_fn=collate_fn, num_workers=12 )

        # train_data_loader_fine = DataLoader(fineTune_dataset, shuffle=True,
        #                             batch_size=config.batch_size, drop_last=False,
        #                             collate_fn=collate_fn, num_workers=12 )

        valid_data_loader_fine = DataLoader(valid_fineTune_dataset, shuffle=True,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 )
        
        for data in train_data_loader_fine:
            count = Counter(data["class_label"].squeeze().numpy())
            print(f'AGNs:{count[0]}, SNIa:{count[1]}, Variables:{count[2]}, SNII:{count[3]}')
      
        # all the three models 
        # print("Initialing The models...")
        # sbert = SBERT( hidden=config.hidden_size, n_layers=config.layers,
        #             attn_heads=config.attn_heads, dropout=config.dropout)

        # # The trainers for each model 
        # print("Creating Downstream Task Trainer...")
        # trainer = SBERTFineTuner(sbert, config.num_classes,
        #                         train_dataloader=train_data_loader,
        #                         valid_dataloader=valid_data_loader,
        #                         lr=config.learning_rate,fold=fold, 
        #                         modelId=0, weights=class_weigth_train)

        # print("Training ConvBERT...")
        # OAAccuracy = 0
        # for epoch in range(config.epochs):
        #     # Train first model
        #     train_OA, _, valid_OA, _ = trainer.train(epoch, config.epochs)
        #     if OAAccuracy < valid_OA:
        #         OAAccuracy = valid_OA
        #         trainer.save(epoch, config.finetune_path)

        # print("Loading pre-trained model parameters...")
        # sbert1 = SBERT( hidden=config.hidden_size, n_layers=config.layers,
        #             attn_heads=config.attn_heads, dropout=config.dropout)
        # sbert_path = config.finetune_path + f"checkpoint-Model-0-fold-{fold}.bert.pth"  
        # sbert1.load_state_dict(torch.load(sbert_path))
        # for param in sbert1.parameters():
        #     param.requires_grad = False
        
        # print("Creating Downstream Task Fine Tuner...")
        # trainer1 = SBERTFineTuner(sbert1, config.num_classes,
        #                         train_dataloader=train_data_loader_fine,
        #                         valid_dataloader=valid_data_loader_fine,
        #                         lr=config.learning_rate,fold=fold,
        #                          modelId=1, preTrain=True, weights=class_weigth_fine)
        
        # print("Fine-tuning ConvBERT...")
        # OAAccuracy = 0
        # for epoch in range(config.epochs-20):
        #     # Train first model
        #     train_OA, _, valid_OA, _ = trainer1.train(epoch, config.epochs)
        #     if OAAccuracy < valid_OA:
        #         OAAccuracy = valid_OA
        #         trainer1.save(epoch, config.finetune_path)

        # print("\n\n\n")
        # print("Testing The models...")
        # trainer1.load(config.finetune_path)
        # OA, Kappa, AA, f1 = trainer1.test(valid_data_loader_fine, "AllSN")
        # df_accuracies_All = df_accuracies_All.append(pd.DataFrame.from_dict({'Model_Name':[f'Model-1-Fold-{fold}'],
        #     'OAccuracy':[OA], 'AAccuracy':[AA], 'F1Score':[f1]}))

        # df_accuracies_All.to_csv(config.accur_path, index=False)

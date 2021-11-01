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
        default='/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/TrainLogsGroups-64-07/CSVStats/',
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
        default="/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints/TrainGroups-64-07/",
        type=str,
        required=False,
        help="The storage path of the pre-trained model.",
    )
    parser.add_argument(
        "--finetune_path",
        default='/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints/TrainGroups-64-07/',
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
        default=64,
        type=int,
        help="",
    )
    parser.add_argument(
        "--hidden_size",
        default=[64, 8, 9, 9, 99//3],
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
    ids_label_dict = get_id_label('/home/barrage/anass/sdss/master_data.fits')
    AGNs = getKeysByValue(ids_label_dict,'AGN')
    Variable = getKeysByValue(ids_label_dict,'Variable')
    SNII = getKeysByValue(ids_label_dict,'TrainSNII')
    SNIa =getKeysByValue(ids_label_dict,'TrainSNIa')
    SpectoSNIa =getKeysByValue(ids_label_dict,'ValidSNIa')
    SpectoSNII = getKeysByValue(ids_label_dict,'ValidSNII')
    random.shuffle(AGNs)
    random.shuffle(Variable)
    random.shuffle(SNII)
    random.shuffle(SNIa)
    random.shuffle(SpectoSNIa)
    random.shuffle(SpectoSNII)

    # print(f'this is agn shape {AGNs}')
    TrainAGN = AGNs[:int(len(AGNs) * .90)]
    TestAGN = AGNs[int(len(AGNs) * .90):]

    TrainVariable = Variable[:int(len(Variable) * .90)]
    TestVariable = Variable[int(len(Variable) * .90):]

    TrainSNII =  SNII[:int(len(SNII) * .90)]
    TestSNII = SNII[int(len(SNII) * .90):]

    TrainSNs = TrainSNII + SNIa + SpectoSNIa[:int(len(SpectoSNIa)/2)] + SpectoSNII[:int(len(SpectoSNII)/2)]

    All_data = TrainAGN + TrainVariable + TrainSNs


    test_data =TestAGN + TestVariable + SpectoSNIa[int(len(SpectoSNIa)/2):] + SpectoSNII[int(len(SpectoSNII)/2):] + TestSNII

    test_data2 =TestAGN + TestVariable + SpectoSNIa[int(len(SpectoSNIa)/2):] + SpectoSNII[int(len(SpectoSNII)/2):]

    test_data3 =TestAGN + TestVariable + SpectoSNIa[int(len(SpectoSNIa)/2):] + TestSNII


    test_data.remove(13651)
    test_data2.remove(13651)
    test_data3.remove(13651)

    # Configuration options
    k_folds = 4

#   Dataframes to store the stats 
    accuu = {'Model_Name':[], 'OAccuracy':[], 'AAccuracy':[], 'F1Score':[]}
    deep_accu = {'Fold':[], 'AvgOAccuracy':[],
        'VarOAccuracy':[], 'AvgAAccuracy':[], 'VarAAccuracy':[],'AvgF1Score':[],'VarF1Score':[] }


    df_accuracies_All =  pd.DataFrame.from_dict(accuu)
    df_Deep_accuracies_All =  pd.DataFrame.from_dict(deep_accu)

    df_accuracies_Conf =  pd.DataFrame.from_dict(accuu)
    df_Deep_accuracies_Conf =  pd.DataFrame.from_dict(deep_accu)

    df_accuracies_Unconf =  pd.DataFrame.from_dict(accuu)
    df_Deep_accuracies_Unconf =  pd.DataFrame.from_dict( deep_accu)

  
    torch.manual_seed(42)

    classes = {'AGN':0,  
     'SNIa':1, 'SNIa?':1, 'zSNIa':1, 'pSNIa':1, 
     'Variable':2, 
     'pSNIbc':3, 'zSNII':3, 'zSNIbc':3 ,'SLSN':3,  'SNIb':3,'pSNII':3, 'SNIc':3,'SNII':3}

    print("Loading Data sets...")
    train_dataset = FinetuneDataset(config.file_path, config.labels_path,
     classes=classes, seq_len=config.max_length, list_IDs=All_data, Transform=True)

    test_dataset = FinetuneDataset(config.file_path, config.labels_path, 
    classes=classes, seq_len=config.max_length, list_IDs=test_data)

    test_dataset2 = FinetuneDataset(config.file_path, config.labels_path, 
    classes=classes, seq_len=config.max_length, list_IDs=test_data2)

    test_dataset3 = FinetuneDataset(config.file_path, config.labels_path, 
    classes=classes, seq_len=config.max_length, list_IDs=test_data3)
    print("training samples: %d, testing samples: %d" %
        (train_dataset.TS_num, test_dataset.TS_num))

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):


        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # collecting stats
      
        accuracies_All = pd.DataFrame.from_dict(accuu)
        accuracies_Conf =  pd.DataFrame.from_dict(accuu)
        accuracies_Unconf =  pd.DataFrame.from_dict(accuu)


        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
  
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)
        
        print("Creating Dataloader...")
        train_data_loader = DataLoader(train_dataset,sampler=train_subsampler,
                                    batch_size=config.batch_size, drop_last=False,
                                        collate_fn=collate_fn, num_workers=12 )
        valid_data_loader = DataLoader(train_dataset, sampler=valid_subsampler,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 )
        
        test_data_loader = DataLoader(test_dataset, shuffle=False,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 )
        test_data_loader2 = DataLoader(test_dataset2, shuffle=False,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 ) 
        test_data_loader3 = DataLoader(test_dataset3, shuffle=False,
                                    batch_size=config.batch_size, drop_last=False,
                                    collate_fn=collate_fn, num_workers=12 ) 


        # all the tHree models 

        print("Initialing The models...")
        sbert = SBERT( hidden=config.hidden_size, n_layers=config.layers,
                    attn_heads=config.attn_heads, dropout=config.dropout)

        sbert1 = SBERT( hidden=config.hidden_size, n_layers=config.layers,
                    attn_heads=config.attn_heads, dropout=config.dropout)

        sbert2 = SBERT( hidden=config.hidden_size, n_layers=config.layers,
                    attn_heads=config.attn_heads, dropout=config.dropout)

      
        
#       The trainers for each model 
        print("Creating Downstream Task Trainer...")
        trainer = SBERTFineTuner(sbert, config.num_classes,
                                train_dataloader=train_data_loader,
                                valid_dataloader=valid_data_loader,lr=config.learning_rate,fold=fold, modelId=0)
        
        trainer1 = SBERTFineTuner(sbert, config.num_classes,
                                train_dataloader=train_data_loader,
                                valid_dataloader=valid_data_loader,lr=config.learning_rate,fold=fold,modelId=1)

        trainer2 = SBERTFineTuner(sbert, config.num_classes,
                                train_dataloader=train_data_loader,
                                valid_dataloader=valid_data_loader,lr=config.learning_rate,fold=fold,modelId=2)
        

        print("Fine-tuning ConvBERT...")
        OAAccuracy = 0
        OAAccuracy1 = 0
        OAAccuracy2 = 0

        for epoch in range(config.epochs):
            # Train first model
            train_OA, _, valid_OA, _ = trainer.train(epoch, config.epochs)
            if OAAccuracy < valid_OA:
                OAAccuracy = valid_OA
                trainer.save(epoch, config.finetune_path)
            
            # Train second model
            train_OA1, _, valid_OA1, _ = trainer1.train(epoch, config.epochs)
            if OAAccuracy1 < valid_OA1:
                OAAccuracy1 = valid_OA1
                trainer1.save(epoch, config.finetune_path)

            # Train Third model
            train_OA2, _, valid_OA2, _ = trainer2.train(epoch, config.epochs)
            if OAAccuracy2 < valid_OA2:
                OAAccuracy2 = valid_OA2
                trainer2.save(epoch, config.finetune_path)



        print("\n\n\n")
        print("Testing The models...")

        for id, model in enumerate([trainer, trainer1, trainer2]):
           
            model.load(config.finetune_path)
            OA, Kappa, AA, f1 = model.test(test_data_loader, "AllSN")
            accuracies_All = accuracies_All.append(pd.DataFrame.from_dict({'Model_Name':[f'Model-{id}-Fold-{fold}'],
             'OAccuracy':[OA], 'AAccuracy':[AA], 'F1Score':[f1]}))

            OA2, Kappa2, AA2, f12 = model.test(test_data_loader2, "ConfSN")
            accuracies_Conf=accuracies_Conf.append(pd.DataFrame.from_dict({'Model_Name':[f'Model-{id}-Fold-{fold}'], 
            'OAccuracy':[OA2], 'AAccuracy':[AA2], 'F1Score':[f12]}))

            OA3, Kappa3, AA3, f13 = model.test(test_data_loader3, "UnconfSN")
            accuracies_Unconf= accuracies_Unconf.append(pd.DataFrame.from_dict({'Model_Name':[f'Model-{id}-Fold-{fold}'],
             'OAccuracy':[OA3], 'AAccuracy':[AA3], 'F1Score':[f13]}))


        df_deep = [df_Deep_accuracies_All, df_Deep_accuracies_Conf, df_Deep_accuracies_Unconf]
        df_norm = [df_accuracies_All, df_accuracies_Conf, df_accuracies_Unconf ]
        _nrom =  [accuracies_All, accuracies_Conf, accuracies_Unconf ]
        df_name = ['All', 'COnf', 'Unconf']

        for index, (df_deep_it, df_norm_it, _nrom_it, df_name_it) in enumerate(zip(df_deep, df_norm,_nrom, df_name)):
            df_deep_it = df_deep_it.append(pd.DataFrame.from_dict({'Fold':[f'Fold-{fold}'],
            'AvgOAccuracy':[np.mean(_nrom_it['OAccuracy'])],'VarOAccuracy':[np.var(_nrom_it['OAccuracy'])], 
            'AvgAAccuracy':[np.mean(_nrom_it['AAccuracy'])], 'VarAAccuracy':[np.var(_nrom_it['AAccuracy'])],
            'AvgF1Score':[np.mean(_nrom_it['F1Score'])],'VarF1Score':[np.var(_nrom_it['F1Score'])] }))
            df_norm_it = df_norm_it.append(_nrom_it)
            df_deep_it.to_csv(config.accur_path +'df_Deep_accuracies_'+df_name_it+'.csv', index=False)
            df_norm_it.to_csv(config.accur_path +'df_accuracies_'+df_name_it+'.csv', index=False)

      


    # Save the avg of all models 

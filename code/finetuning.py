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
        "--labels_path",
        default='/home/barrage/anass/sdss/master_data.fits',
        type=str,
        required=False,
        help="The labels path.",
    )
    parser.add_argument(
        "--pretrain_path",
        default="/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints/finetune-for-cal/",
        type=str,
        required=False,
        help="The storage path of the pre-trained model.",
    )
    parser.add_argument(
        "--finetune_path",
        default='/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints/finetune-for-cal/',
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
        default=70,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="",
    )
    parser.add_argument(
        "--hidden_size",
        default=[512, 8, 9, 9, 99//3],
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
        default=3e-4,
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

    # classes = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:9, 12:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16}
    # classes = {'AGN':0, 'SLSN':1, 'SNII':2, 'SNIa':3, 'SNIa?':4, 'SNIb':5, ' ':6, 
    # 'Unknown':7, 'Variable':8, 'pSNII':9, 'pSNIa':10, 'pSNIbc':11, 'zSNII':12,
    #  'zSNIa':13, 'zSNIbc':14}

    #classes = {'AGN':0, 'SLSN':1, 'SNII':1, 'SNIa':1, 'SNIa?':1, 'SNIb':1, 'SNIc':1, 
    #'Variable':2, 'pSNII':1, 'pSNIa':1, 'pSNIbc':1, 'zSNII':1,
    # 'zSNIa':1, 'zSNIbc':1}
   
    
    # Generating data loaders 
    ids_label_dict = get_id_label('/home/barrage/anass/sdss/master_data.fits')
    AGNs = getKeysByValue(ids_label_dict,'AGN')
    Variable = getKeysByValue(ids_label_dict,'Variable')
    SNII =getKeysByValue(ids_label_dict,'TrainSNII')
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
    k_folds = 5

    # For fold results
    resultsOA = []
    resultsF1 = []
    resultsAA = []
    resultsKappa = []

    resultsOA2 = []
    resultsF12 = []
    resultsAA2 = []
    resultsKappa2 = []


    resultsOA3 = []
    resultsF13 = []
    resultsAA3 = []
    resultsKappa3 = []

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


        

        print("Initialing SITS-BERT...")
        sbert = SBERT( hidden=config.hidden_size, n_layers=config.layers,
                    attn_heads=config.attn_heads, dropout=config.dropout)

        # if config.pretrain_path is not None:
        #     print("Loading pre-trained model parameters...")
        #     sbert_path = config.pretrain_path + "checkpoint.bert.pth"
        #     sbert.load_state_dict(torch.load(sbert_path))

        

        print("Creating Downstream Task Trainer...")
        trainer = SBERTFineTuner(sbert, config.num_classes,
                                train_dataloader=train_data_loader,
                                valid_dataloader=valid_data_loader,lr=config.learning_rate,fold=fold)

        

        print("Fine-tuning ConvBERT...")
        OAAccuracy = 0
        for epoch in range(config.epochs):
            train_OA, _, valid_OA, _ = trainer.train(epoch, config.epochs, fold)
            if OAAccuracy < valid_OA:
                OAAccuracy = valid_OA
                trainer.save(epoch, config.finetune_path)

        print("\n\n\n")
        print("Testing ConvBERT...")
        trainer.load(config.finetune_path)
        OA, Kappa, AA, f1 = trainer.test(test_data_loader,fold, "AllSN")
        OA2, Kappa2, AA2, f12 = trainer.test(test_data_loader2,fold, "ConfSN")
        OA3, Kappa3, AA3, f13 = trainer.test(test_data_loader3,fold, "UnconfSN")

        print('test_OA = %.2f, test_kappa = %.3f, test_AA = %.3f, test_F1 = %.3f' % (OA, Kappa, AA, f1))
        resultsAA.append(AA)
        resultsF1.append(f1)
        resultsKappa.append(Kappa)
        resultsOA.append(OA)

        resultsAA2.append(AA2)
        resultsF12.append(f12)
        resultsKappa2.append(Kappa2)
        resultsOA2.append(OA2)

        resultsAA3.append(AA3)
        resultsF13.append(f13)
        resultsKappa3.append(Kappa3)
        resultsOA3.append(OA3)


    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')

    print('--------------------------------\n')
    print('All type of SN\n')
    print('Avg_test_OA = %.2f, Avg_test_kappa = %.3f, Avg_test_AA = %.3f, Avg_test_F1 = %.3f' 
    % (sum(resultsOA)/k_folds, sum(resultsKappa)/k_folds, sum(resultsAA)/k_folds,
     sum(resultsF1)/k_folds))

    print('--------------------------------\n')
    print('Only spectro confirmed \n')
    print('Avg_test_OA = %.2f, Avg_test_kappa = %.3f, Avg_test_AA = %.3f, Avg_test_F1 = %.3f' 
    % (sum(resultsOA2)/k_folds, sum(resultsKappa2)/k_folds, sum(resultsAA2)/k_folds,
     sum(resultsF12)/k_folds))


    print('--------------------------------\n')
    print('Only unconfirmed\n')
    print('Avg_test_OA = %.2f, Avg_test_kappa = %.3f, Avg_test_AA = %.3f, Avg_test_F1 = %.3f' 
    % (sum(resultsOA3)/k_folds, sum(resultsKappa3)/k_folds, sum(resultsAA3)/k_folds,
     sum(resultsF13)/k_folds))



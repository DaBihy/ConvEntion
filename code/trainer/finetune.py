import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import fold
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from model import SBERT, SBERTClassification
from .focal_loss import FocalLoss 
from .plastic_loss import loss_fn
import collections
from torch.utils.tensorboard import SummaryWriter
import tensorboard
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import gc
plt.switch_backend('agg')
import itertools

def get_f1_score(confusion_matrix, i):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for j in range(len(confusion_matrix)):
        if (i == j):
            # true positive: 真實為i，預測為i (confusion matrix 中的對角線項目)
            TP += confusion_matrix[i, j]
            tmp = np.delete(confusion_matrix, i, 0)
            tmp = np.delete(tmp, j, 1)
            # true negative: 真實不為i, 預測不為i (confusion matrix 中, row=col=i 以外的項目總合)
            TN += np.sum(tmp)
        else:
            if (confusion_matrix[i, j] != 0):
                # false negative: 真實為i, 預測不為i (confusion matrix中, row i 上不為0的總數)
                FN += confusion_matrix[i, j]
            if (confusion_matrix[j, i] != 0):
                # false positive: 真實不為i, 預測為i (confusion matrix中, col i 上不為0的總數)
                FP += confusion_matrix[j, i]

    recall = TP / (FN + TP)
    precision = TP / (TP + FP)
    f1_score = 2 * 1/(1/recall + 1/precision)
    
    return f1_score


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)


def average_accuracy(matrix):
    correct = np.diag(matrix)
    all = matrix.sum(axis=0)
    accuracy = correct / all
    aa = np.average(accuracy)
    return aa


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', figureName='test', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    save_path = '/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/fineLogsAug-for-cal/'+figureName+'.png'
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def reset_weights(m):

    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class SBERTFineTuner:
    def __init__(self, sbert: SBERT, num_classes: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float = 1e-3, with_cuda: bool = True,
                 cuda_devices=None, log_freq: int = 100, fold=0):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.writer = SummaryWriter(f'/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/fineLogsAug-for-cal/Train-fold-{fold}')
        self.writer1 = SummaryWriter(f'/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/fineLogsAug-for-cal/Valid-fold-{fold}')
        self.fold = fold
        self.sbert = sbert
        gc.collect()
        torch.cuda.empty_cache()
        
        self.model = SBERTClassification(sbert, num_classes).to(self.device)
        # Reset the weight of the model after each fold 
        self.model.apply(reset_weights)

        # calculate the number of parameters in the model 
        param = filter(lambda p: p.requires_grad, self.model.parameters())
        param = sum([np.prod(p.size()) for p in param]) / 1_000_000
        print('Trainable Parameters: %.3fM' % param)
        # init the optimizer for the model  
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.3, amsgrad=True)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, patience=8, factor=0.5, verbose=True)

        preTrain = None
        

        self.num_classes = num_classes

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=[0,1,2,3])

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
       

        self.criterion = FocalLoss()

        if preTrain is not None:
            print("Loading pre-trained model parameters...")
            sbert_path = "/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/checkpoints-2/finetune2checkpoint.tar"
            checkpoint = torch.load(sbert_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.criterion = nn.CrossEntropyLoss()
        self.log_freq = log_freq


    def train(self, epoch, epochs, fold):
        train_loss = 0.0
        counter = 0
        total_correct = 0
        total_element = 0
        class_targets = []
        da_result = []

        matrix = np.zeros([self.num_classes, self.num_classes])
        for data in self.train_dataloader:
            data = {key: value.to(self.device) for key, value in data.items()}

            classification = self.model(data["bert_input"].float(), None, data["bands"].long(),data["ebv"].float() )
            loss = self.criterion(classification, data["class_label"].squeeze().long(), data["weight"].float())
            classification_target = data["class_label"].squeeze()

            if epoch == epochs-1:
                print(f'This is the shape {classification_target.shape}')
                class_targets.append(classification_target.cpu().detach().numpy())
                da_result.append(classification.cpu().detach().numpy())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss += loss.item()

            classification_result = classification.argmax(dim=-1)
            correct = classification_result.eq(classification_target).sum().item()

            total_correct += correct
            total_element += data["class_label"].nelement()

           

            for row, col in zip(classification_result, classification_target):
                matrix[row, col] += 1

            counter += 1
        
        train_loss /= counter
        train_OA = total_correct / total_element * 100
        train_kappa = kappa(matrix)
        lr=self.optim.param_groups[0]["lr"]


        valid_loss, valid_OA, valid_kappa = self._validate()
        #self.scheduler.step(valid_loss)

        np.save(f'/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/Outputs/targets_Train-fold-{fold}.npy', np.array(class_targets), allow_pickle=True)
        np.save(f'/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/Outputs/logits_Train-fold-{fold}.npy', np.array(da_result), allow_pickle=True)
        print("EP%d, train_loss=%.3f, train_OA=%.2f, train_Kappa=%.3f, validate_loss=%.3f, validate_OA=%.2f, validate_Kappa=%.3f"
              % (epoch,train_loss, train_OA, train_kappa,valid_loss, valid_OA, valid_kappa))

        self.writer.add_scalar('Loss', train_loss, global_step=epoch)
        self.writer1.add_scalar('Loss', valid_loss, global_step=epoch)
        self.writer.add_scalar('OAccuracy', train_OA, global_step=epoch)
        self.writer1.add_scalar('OAccuracy', valid_OA, global_step=epoch)
        #self.writer.add_scalar('Lr', lr, global_step=epoch)



        return train_OA, train_kappa, valid_OA, valid_kappa

    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            total_correct = 0
            total_element = 0
            matrix = np.zeros([self.num_classes, self.num_classes])
            for data in self.valid_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                classification = self.model(data["bert_input"].float(), None, data["bands"].long(),data["ebv"].float() )

                loss = self.criterion(classification, data["class_label"].squeeze().long(), data["weight"].float())

                valid_loss += loss.item()

                classification_result = classification.argmax(dim=-1)
                classification_target = data["class_label"].squeeze()

                correct = classification_result.eq(classification_target).sum().item()
                total_correct += correct
                total_element += data["class_label"].nelement()
                for row, col in zip(classification_result, classification_target):
                    matrix[row, col] += 1

                counter += 1

            valid_loss /= counter
            valid_OA = total_correct / total_element * 100
            valid_kappa = kappa(matrix)

        self.model.train()

        return valid_loss, valid_OA, valid_kappa

    def test(self, data_loader, fold, type):
        with torch.no_grad():
            self.model.eval()

            total_correct = 0
            total_element = 0
            class_targets = []
            class_resuls = []
            da_result = []
            matrix = np.zeros([self.num_classes, self.num_classes])
            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}

                result = self.model(data["bert_input"].float(),
                                    None, data["bands"].long(),data["ebv"].float() )
                da_result.append(result.cpu().detach().numpy())
                classification_result = result.argmax(dim=-1)
                classification_target = data["class_label"].squeeze()

                class_targets.append(classification_target.cpu().detach().numpy())

                # ytrue = classification_target.cpu()
                # ypred = classification_result.cpu()
                correct = classification_result.eq(classification_target).sum().item()
                # Here 
                # conf_mat = confusion_matrix(ytrue.view(-1), ypred.view(-1), normalize='true')
                # print(conf_mat)
                total_correct += correct
                total_element += data["class_label"].nelement()
                for row, col in zip(classification_target,classification_result):
                    matrix[row, col] += 1
            # print(f'The matrix shape is {conf_mat.shape}')
            # print(matrix)
            f1_score = 0
            for i in range(self.num_classes):
                f1_score += get_f1_score(matrix,i)
            f1_score/=self.num_classes    
            test_OA = total_correct * 100.0 / total_element
            test_kappa = kappa(matrix)
            test_AA = average_accuracy(matrix)

            # Save the reults to file 
            print(f"the number of target smaples is after classification {len(class_targets)}")
            print(f"the number of results smaples is after classification {len(class_resuls)}")

            np.save(f'/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/Outputs/targets_Test-fold-{fold}.npy', np.array(class_targets), allow_pickle=True)
            np.save(f'/home/barrage/anass/ConvBERTSparseSDSSResultsRegulWeighted/Outputs/logits_Test-fold-{fold}.npy', np.array(da_result), allow_pickle=True)
            # np.save('/home/anass/plasticc/ConvBERTBandSep3DCNN/class_resuls1.npy', np.array(class_resuls), allow_pickle=True)

            # Plot the conf matrixes 

            fig = plt.figure(figsize=(10,10))
            # plt.imshow(conf_mat)
            # dclasses = np.array(['AGN', 'SLSN', 'SNII', 'SNIa', 'SNIa?', 'SNIb', 'SNIc', 'Variable', 'pSNII', 'pSNIa', 'pSNIbc', 'zSNII','zSNIa', 'zSNIbc'])
            dclasses = np.array(['AGN', 'SNIa', 'Variable', 'SNautre'])
            #dclasses = np.array(['AGN', 'SN', 'Variable'])

            plot_confusion_matrix(matrix, dclasses, normalize=True, figureName=f'FourClassesPercent-fold-{fold}-{type}')
            self.writer.add_figure(f'FourClassesPercent-fold-{fold}', fig)
            self.writer.close()

            plot_confusion_matrix(matrix, dclasses, normalize=False, figureName=f'FourClassesNumber-fold-{fold}-{type}')
      


        self.model.train()
        
        return test_OA, test_kappa, test_AA, f1_score

    def save(self, epoch, file_path):
        output_path = file_path +  f"checkpoint-fold-{self.fold}.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path):
        input_path = file_path + f"checkpoint-fold-{self.fold}.tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model loaded from:" % epoch, input_path)
        return input_path

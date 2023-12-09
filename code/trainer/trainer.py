import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import ConvBERTClassification
from .focal_loss import FocalLoss
import matplotlib.pyplot as plt
import gc
import wandb

plt.switch_backend('agg')
import itertools

def get_f1_score(confusion_matrix, i):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for j in range(len(confusion_matrix)):
        if (i == j):
            # true positive
            TP += confusion_matrix[i, j]
            tmp = np.delete(confusion_matrix, i, 0)
            tmp = np.delete(tmp, j, 1)
            # true negative:
            TN += np.sum(tmp)
        else:
            if (confusion_matrix[i, j] != 0):
                # false negative: 
                FN += confusion_matrix[i, j]
            if (confusion_matrix[j, i] != 0):
                # false positive
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
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    wandb.log({figureName: wandb.Image(plt)})
    # plt.savefig(save_path)
    plt.close()


def reset_weights(m):

    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class ConvBERTTrainer:
    def __init__(self, sbert: SBERT, sbertModel: SBERT, num_classes: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float = 1e-3, with_cuda: bool = True,
                 momentum=0.9, log_freq: int = 100, fold=0, modelId=None, reset=True, file_path=None):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(f"cuda" if cuda_condition else "cpu")
        self.fold = fold
        self.sbert = sbert
        self.sbertModel = sbertModel
        self.modelId = modelId
        self.reset = reset
        self.m = momentum
        self.log_freq = log_freq
        self.num_classes = num_classes
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader


        gc.collect()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
    
        self.model = ConvBERTClassification(sbertModel, num_classes).to(self.device)

        # Reset the weight of the model after each fold
        if self.reset==True: 
            print("reseting weights...")
            self.model.apply(reset_weights)

        # init the optimizer for the model  
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.1339, amsgrad=True)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=[0,1])

        self.criterion = FocalLoss()
       


    def train(self):
        train_loss = 0.0
        counter = 0
        total_correct = 0
        total_element = 0

        matrix = np.zeros([self.num_classes, self.num_classes])
        for data in self.train_dataloader:
            data = {key: value.to(self.device) for key, value in data.items()}

        
            model_view_1 = self.model(data["bert_input"].float(), None, data["bands"].long())
            loss = self.criterion(model_view_1, data["class_label"].squeeze().long()) 
        
            loss = loss.mean()

            classification_target = data["class_label"].squeeze()
     
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            train_loss += loss.item()

            classification_result = model_view_1.argmax(dim=-1)
            correct = classification_result.eq(classification_target).sum().item()

            total_correct += correct
            total_element += data["class_label"].nelement()

            for row, col in zip(classification_result, classification_target):
                matrix[row, col] += 1

            counter += 1
        
        train_loss /= counter
        train_OA = total_correct / total_element * 100
        train_kappa = kappa(matrix)

        valid_loss, valid_OA, valid_kappa = self._validate()
       
        wandb.log({f'F{self.fold}/M{self.modelId}/Loss': train_loss,
                    f'F{self.fold}/M{self.modelId}/Val_loss': valid_loss, 
                    f'F{self.fold}/M{self.modelId}/OAccuracy': train_OA,
                      f'F{self.fold}/M{self.modelId}/Val_OAccuracy': valid_OA })

      
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
                
               
                model_view_1 = self.model(data["bert_input"].float(), None, data["bands"].long())
                loss = self.criterion(model_view_1, data["class_label"].squeeze().long()) 
        
                loss = loss.mean()

                valid_loss += loss.item()
                classification_result = model_view_1.argmax(dim=-1)
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

    def test(self, data_loader, type, className):
        with torch.no_grad():
            self.model.eval()

            total_correct = 0
            total_element = 0
            class_targets = []
            class_results = []
            matrix = np.zeros([self.num_classes, self.num_classes])
            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}

                result = self.model(data["bert_input"].float(), None, data["bands"].long())
                classification_result = result.argmax(dim=-1)
                classification_target = data["class_label"].squeeze()
                
                class_targets.append(classification_target.cpu().detach().numpy())
                class_results.append(classification_result.cpu().detach().numpy())

              
                correct = classification_result.eq(classification_target).sum().item()
               
                total_correct += correct
                total_element += data["class_label"].nelement()
                for row, col in zip(classification_target,classification_result):
                    matrix[row, col] += 1
           
            f1_score = 0
            for i in range(self.num_classes):
                f1_score += get_f1_score(matrix,i)
            f1_score/=self.num_classes    
            test_OA = total_correct * 100.0 / total_element
            test_kappa = kappa(matrix)
            test_AA = average_accuracy(matrix)

            res_class = np.concatenate(class_results)
            target_class = np.concatenate(class_targets)

            wandb.log({"roc" : wandb.plot.roc_curve(target_class, res_class,
                        labels=None, classes_to_plot=None)})
            fig = plt.figure(figsize=(10,10))
           

        self.model.train()
        
        return test_OA, test_kappa, test_AA, f1_score

    def save(self, epoch, file_path):
        output_path = file_path +  f"check-M-{self.modelId}-f{self.fold}.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        bert_path = file_path + f"check-M-{self.modelId}-f{self.fold}.bert.pth"  
        torch.save(self.sbertModel.state_dict(), bert_path)

        return output_path

    def load(self, file_path):
        input_path =  file_path +  f"check-M-{self.modelId}-f{self.fold}.tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model loaded from:" % epoch, input_path)
        return input_path
    
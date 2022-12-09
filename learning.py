import os
os.environ["http_proxy"] = "http://proxy.cmu.edu:3128"
os.environ["https_proxy"] = "https://proxy.cmu.edu:3128"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import librosa
import numpy as np
import sklearn
import sklearn.metrics
import gc
import zipfile
import pandas as pd
import random
from tqdm.autonotebook import tqdm
import os
import datetime
import torch.nn as nn
import torch
from adamp import AdamP
from utlis import calculate_threshold
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device: ", device)

root = "./data/wav_resampled16kHz"

import wandb
wandb.login(key="ca643ee3068ccf1ee582f09717233a3735298ed0")

run = wandb.init(
    name = "baseline_model_no_dropout", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "11685_project", ### Project should be created in your wandb account 
)

# Dataset class to load train and validation data

from calendar import day_abbr
from sklearn.preprocessing import normalize

config = {
    'epochs': 100,
    'batch_size' : 256,
    'context' : 15,
    'learning_rate' : 0.05,
    'architecture' : 'baseline_model'
    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
}

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, context, offset=0, partition= "train", limit=-1, shuffle=True, Normalization=True, mfcc_list=["mfcc", "mfcc_320", "mfcc_480"]): # Feel free to add more arguments

        self.context = context
        self.offset = context
        self.data_path = data_path
        self.normalization = Normalization
        self.mfcc_list = mfcc_list
        
        self.mfcc_list = []
        self.transcripts = []
        for index, mfcc_file in enumerate(mfcc_list):
            if partition == "train":
                self.mfcc_dir = data_path + '/' + partition + "/" + mfcc_file + "/"
                self.transcript_dir = data_path + '/' + partition + "/labels/"
            else:
                self.mfcc_dir = data_path + '/' + partition + "/" + mfcc_file + "/"
                self.transcript_dir = data_path + '/' + partition + "/labels/"

            mfcc_names = sorted(os.listdir(self.mfcc_dir))
            transcript_names = sorted(os.listdir(self.transcript_dir))
            assert len(mfcc_names) == len(transcript_names)

            self.mfccs = []

            for i in range(0, len(mfcc_names)):
    
                mfcc = np.load(self.mfcc_dir + mfcc_names[i])
                self.mfccs.append(mfcc)
            #   Optionally do Cepstral Normalization of mfcc
            #   Load the corresponding transcript
                if index == 0:
                    transcript = np.load(self.transcript_dir + transcript_names[i])
                    self.transcripts.append(transcript)

            # if partition == "train":
            #     if shuffle == True:
            #         Pairs = list(zip(self.mfccs, self.transcripts))
            #         random.shuffle(Pairs)
            #         self.mfccs, self.transcripts = zip(*Pairs)

            # Each mfcc is of shape T1 x 20, T2 x 20, ...
            # Each transcript is of shape (T1+2) x 20, (T2+2) x 20

            # TODO: Concatenate all mfccs in self.X such that the final shape is T x 20 (Where T = T1 + T2 + ...) 
            self.mfccs = np.concatenate(self.mfccs, axis=0)
            self.length = len(self.mfccs)
            # TODO: Concatenate all transcripts in self.Y such that the final shape is (T,) meaning, each time step has one phoneme output
            if index == 0:
                self.transcripts = np.concatenate(self.transcripts, axis=0)
            # Hint: Use numpy to concatenate
            # Take some time to think about what we have done. self.mfcc is an array of the format (Frames x Features). Our goal is to recognize phonemes of each frame
            # From hw0, you will be knowing what context is. We can introduce context by padding zeros on top and bottom of self.mfcc
            if context != 0:
                zero_paddings = np.zeros((context, 20))
                up_paded = np.vstack((zero_paddings, self.mfccs))
                down_paded = np.vstack((up_paded, zero_paddings))
                self.mfccs = down_paded
                
            self.mfcc_list.append(self.mfccs)
        self.mfccs = np.stack(self.mfcc_list)


    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        
        start_index = ind + self.offset - self.context
        
        ## Calculate ending timestep using offset and context (1 line)
        end_index = ind + self.offset + self.context + 1

        frames = self.mfccs[:, start_index:end_index, :]
        
        # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
        # After slicing, you get an array of shape 2*context+1 x 15. But our MLP needs 1d data and not 2d.
        height = frames.shape[0]
        width = frames.shape[1]
        if self.normalization == True:
          frames = frames - frames.mean(axis=0, keepdims=True)
          # frames_variance = np.var(frames, axis=0)
          # frames = np.divide(frames, np.tile(frames_variance, (height, 1)))
        frames = torch.FloatTensor(frames) # Convert to tensors
        onset = torch.tensor(self.transcripts[ind])       
        
        return frames, onset
    
class AudioTestDataset(torch.utils.data.Dataset):
    # TODO: Create a test dataset class similar to the previous class but you dont have transcripts for this
    # Imp: Read the mfccs in sorted order, do NOT shuffle the data here or in your dataloader.
    def __init__(self, data_path, context, offset=0, limit=-1, Normalization=True, mfcc_list=["mfcc", "mfcc_320", "mfcc_480"]): # Feel free to add more arguments

        self.context = context
        self.offset = context
        self.data_path = data_path
        self.mfcc_dir = data_path + '/' + "test" + "/mfcc/"
        self.normalization = Normalization
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.mfcc_list = mfcc_list
        
        self.mfcc_list = []
        for mfcc_file in mfcc_list:
            
            self.mfccs= []
            for i in range(0, len(mfcc_names)):
            #   Load a single mfcc
                mfcc = np.load(self.mfcc_dir + mfcc_names[i])
                self.mfccs.append(mfcc)

            # NOTE:
            # Each mfcc is of shape T1 x 20, T2 x 20, ...
            # Each transcript is of shape (T1+2) x 20, (T2+2) x 20 before removing [SOS] and [EOS]

            self.mfccs = np.concatenate(self.mfccs, axis=0)
            self.length = len(self.mfccs)

            # Take some time to think about what we have done. self.mfcc is an array of the format (Frames x Features). Our goal is to recognize phonemes of each frame
            # From hw0, you will be knowing what context is. We can introduce context by padding zeros on top and bottom of self.mfcc
            if context != 0:
                zero_paddings = np.zeros((context, 20))
                up_paded = np.vstack((zero_paddings, self.mfccs))
                down_paded = np.vstack((up_paded, zero_paddings))
                self.mfccs = down_paded
            
            self.mfcc_list.append(self.mfccs)
        self.mfccs = np.stack(self.mfcc_list)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        start_index = ind + self.offset - self.context
        ## Calculate ending timestep using offset and context (1 line)
        end_index = ind + self.offset + self.context + 1
        frames = self.mfccs[:, start_index:end_index, :]
        height = frames.shape[0]
        width = frames.shape[1]
        if self.normalization == True:
          frames = frames - frames.mean(axis=0, keepdims=True)

        frames = torch.FloatTensor(frames) # Convert to tensors 

        return frames

data_path = root
train_data = AudioDataset(data_path, context = config['context'], offset=0, partition= "train", limit=-1)
val_data = AudioDataset(data_path, context = config['context'], offset=0, partition= "dev", limit=-1, shuffle=False) 

train_loader = torch.utils.data.DataLoader(train_data, num_workers= 4,
                                           batch_size=config['batch_size'], pin_memory= True,
                                           shuffle= True)

val_loader = torch.utils.data.DataLoader(val_data, num_workers= 2,
                                         batch_size=config['batch_size'], pin_memory= True,
                                         shuffle= False)

print("Batch size: ", config['batch_size'])
print("Context: ", config['context'])
print("Input size: ", (2*config['context']+1)*20)

print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

class Network(torch.nn.Module):

    def __init__(self, context):

        super(Network, self).__init__()

        input_size = 3 #Why is this the case?
        num_classes = 2
        dropout_rate = 0
        self.conv1 = torch.nn.Conv2d(input_size, 10, kernel_size=(5, 3))
        self.bn1 = torch.nn.BatchNorm2d(num_features=10)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm2d(num_features=20)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1, 2))
        self.flatten = torch.nn.Flatten()
        
        # self.backbone = torch.nn.Sequential(
        #     torch.nn.Conv2d(input_size, 10, kernel_size=(5, 3)),
        #     torch.nn.BatchNorm1d(num_features=10),
        #     torch.nn.MaxPool2d(kernel_size=(1, 3)),
        #     torch.nn.Conv2d(10, 20, kernel_size=3), 
        #     torch.nn.BatchNorm1d(num_features=20),
        #     torch.nn.MaxPool2d(kernel_size=(1, 3)),
        #     torch.nn.Flatten()
        # )
        
        self.cls_layer = nn.Sequential(
            torch.nn.Linear(in_features=1000, out_features=256),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_features=256, out_features=num_classes)
        )          

    def forward(self, x):
        feats = self.conv1(x)
        feats = self.bn1(feats)
        feats = self.pool1(feats)
        feats = self.conv2(feats)
        feats = self.bn2(feats)
        feats = self.pool2(feats)
        feats = self.flatten(feats)
        
        out = self.cls_layer(feats)
        return out

model = Network(config['context']).to(device)
frames,onsets = next(iter(train_loader))

use_amp = True
criterion = torch.nn.CrossEntropyLoss() #Defining Loss function 
optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
# optimizer = AdamP(model.parameters(), lr=config['learning_rate']) #Defining Optimizer
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25,30], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(model, optimizer, criterion, dataloader, epoch):

    model.train()
    total_train_loss = 0.0 #Monitoring Loss
    
    phone_true_list = []
    phone_pred_list = []
    phone_pred_score_list = []

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    
    for i , (mfccs, onsets) in enumerate(dataloader):
        onsets = torch.squeeze(onsets, 1).long()
        mfccs = mfccs.to(device)
        onsets = onsets.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            ### Forward Propagation
            logits = model(mfccs)
            ### Loss Calculation
            loss = criterion(logits, onsets)

        train_loss = loss.item()
        
        

        # train_acc = float(torch.sum(logits.argmax(axis=1) == onsets) / onsets.shape[0])
        
        total_train_loss += loss.item()
        
        ### Get Predictions
        predicted_score = logits[:, 1]
        phone_true_list.extend(onsets.cpu().tolist())
        phone_pred_score_list.extend(predicted_score.cpu().tolist())

        # total_train_acc += float(torch.sum(logits.argmax(axis=1) == onsets) / onsets.shape[0])

    #     batch_bar.set_postfix(
    # acc="{:.04f}%".format(train_acc),
    # loss="{:.04f}".format(train_loss),
    # lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        ### Initialize Gradients
        optimizer.zero_grad()

        ### Backward Propagation
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        ### Gradient Descent
        scaler.update()

        batch_bar.update()
  
    batch_bar.close()
    total_train_loss /= len(dataloader)
    
    threshold = calculate_threshold(phone_true_list, phone_pred_score_list)
    predicted_phonemes = phone_pred_score_list >= threshold
    
    accuracy = sklearn.metrics.accuracy_score(phone_true_list, predicted_phonemes) 
    auc = sklearn.metrics.roc_auc_score(phone_true_list, phone_pred_score_list)
    auprc = sklearn.metrics.average_precision_score(phone_true_list, phone_pred_score_list)
    f1 = sklearn.metrics.f1_score(phone_true_list, predicted_phonemes)
    recall = sklearn.metrics.recall_score(phone_true_list, predicted_phonemes)
    precision = sklearn.metrics.precision_score(phone_true_list, predicted_phonemes)
    
    return train_loss, optimizer.param_groups[0]["lr"], accuracy, auc, auprc, f1, recall, precision

def eval(model, dataloader):

    model.eval() # set model in evaluation mode
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    phone_true_list = []
    phone_pred_list = []
    phone_pred_score_list = []
    total_val_loss = 0.0

    for i, data in enumerate(dataloader):

        frames, onsets = data
        ### Move data to device (ideally GPU)
        onsets = torch.squeeze(onsets, 1).long()
        frames, onsets = frames.to(device), onsets.to(device) 

        with torch.inference_mode(): # makes sure that there are no gradients computed as we are not training the model now
            ### Forward Propagation
            logits = model(frames)
            loss = criterion(logits, onsets)
        
        
        val_loss = loss.item()

        # val_acc = float(torch.sum(logits.argmax(axis=1) == onsets) / onsets.shape[0])
        
        total_val_loss += loss.item()

        # total_val_acc += float(torch.sum(logits.argmax(axis=1) == onsets) / onsets.shape[0])

    #     batch_bar.set_postfix(
    # acc="{:.04f}%".format(val_acc),
    # loss="{:.04f}".format(val_loss))
        ### Get Predictions
        predicted_score = logits[:, 1]
        phone_true_list.extend(onsets.cpu().tolist())
        phone_pred_score_list.extend(predicted_score.cpu().tolist())
        
        # Do you think we need loss.backward() and optimizer.step() here?
    
        # total_val_loss /= len(dataloader)

        # total_val_acc /= len(dataloader)
        del frames, onsets, logits
        torch.cuda.empty_cache()
        batch_bar.update()
        
    batch_bar.close()
    total_val_loss /= len(dataloader)
    
    threshold = calculate_threshold(phone_true_list, phone_pred_score_list)
    predicted_phonemes = phone_pred_score_list >= threshold

    # total_val_acc /= len(dataloader)
    ### Calculate Accuracy
    accuracy = sklearn.metrics.accuracy_score(phone_true_list, predicted_phonemes) 
    auc = sklearn.metrics.roc_auc_score(phone_true_list, phone_pred_score_list)
    auprc = sklearn.metrics.average_precision_score(phone_true_list, phone_pred_score_list)
    f1 = sklearn.metrics.f1_score(phone_true_list, predicted_phonemes)
    recall = sklearn.metrics.recall_score(phone_true_list, predicted_phonemes)
    precision = sklearn.metrics.precision_score(phone_true_list, predicted_phonemes)
    
    return accuracy, total_val_loss, auc, auprc, f1, recall, precision

torch.cuda.empty_cache()

best_acc = 0.0 ### Monitor best accuracy in your run

for epoch in range(config['epochs']):
    print("\nEpoch {}/{}".format(epoch+1, config['epochs']))

    train_loss, learning_rate, train_acc, train_auc, train_auprc, train_f1, train_recall, train_precision = train(model, optimizer, criterion, train_loader, epoch)

    print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}\tTrain auc".format(
        epoch + 1,
        config['epochs'],
        train_acc,
        train_loss,
        learning_rate,
        train_auc
    ))

    eval_accuracy, eval_loss, eval_auroc, eval_auprc, eval_f1, eval_recall, eval_precision = eval(model, val_loader)
    
    scheduler.step(eval_accuracy)
    
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_auroc": train_auc,
        "train_pre": train_precision,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "eval_auroc": eval_auroc,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_pre": eval_precision
    })

    print("Validation Loss: {:.2f}\tValidation Acc: {:.4f}%\tValidation auc: {:.4f}%\tValidation f1: {:.4f}%".format(eval_loss, eval_accuracy, eval_auroc, eval_f1))


    ### Save checkpoint if accuracy is better than your current best
    if eval_accuracy >= best_acc:

        ### Save checkpoint with information you want
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_loss,
                'acc': eval_accuracy,
                'auroc': eval_auroc,
                'auprc': eval_auprc,
                'f1_score': eval_f1,
                'recall': eval_recall}, 
            './model_checkpoint_base.pth'
        )
      
        best_acc = eval_accuracy
      
run.finish()

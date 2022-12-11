import sys
import subprocess
import codecs
import seaborn as sns
import os
import datetime
from tqdm.notebook import tqdm
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from collections import Counter
from model_m2dcnn import M2DCNN
from model_3dcnn import CNN3D
from dataset import mt_Dataset, lpp_Dataset

import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from torch.optim import Adam 
from torch.optim.lr_scheduler import ExponentialLR


device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

def seed_everything(seed=1234):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_model_m2dcnn(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, is3D=False, schedule=True):
    model.to(device)
    print(device)
    train_acc = []
    valid_acc = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss, epoch_corrects, epoch_acc = 0.0, 0, 0.0
            iteration = 0
            length = len(dataloaders_dict[phase].dataset)
            print(dataloaders_dict)
            for inputs, labels in dataloaders_dict[phase]:
                iteration += 1
                if is3D:
                    inputs = inputs.unsqueeze(1)
                if iteration%100==0:
                    print(f"{iteration*5000/length : .2f}", "%")
                
                
                optimizer.zero_grad()
                inputs = inputs.float()              
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # Backprop
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    batch_loss = loss.item() * inputs.size(0)  
                    epoch_loss += batch_loss
                    epoch_corrects += torch.sum(preds == labels.data)
                
                #print('{} : Minibatch {}/{} finished (Minibatch Loss: {:.4f})'.format(datetime.datetime.now(),min(batch_size*iteration,length),length, batch_loss/batch_size))
        
            epoch_loss = epoch_loss / length
            epoch_acc = epoch_corrects.double() /length
            if phase == 'train':
                train_acc.append([epoch_acc.cpu(),epoch_loss])
            else:
                valid_acc.append([epoch_acc.cpu(),epoch_loss])
            print('##### {} Loss: {:.4f} Acc: {:.4f} #####'.format(phase, epoch_loss, epoch_acc))
            
        if schedule:
            scheduler.step()
        #torch.save(model.state_dict(), save_path)
        
        # Fast stop
        if valid_acc[-1][1] < 0.1:
            print("Stop!")
            return model, train_acc, valid_acc
        
    return model, train_acc, valid_acc

def test_model(model, dataloaders_dict, nb_classes):
    model.eval()
    model.to(device)
    corrects = 0
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for inputs, labels in tqdm(dataloaders_dict["test"]):
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
    plt.figure()
    sns.heatmap(confusion_matrix/torch.sum(confusion_matrix)*10)
    plt.savefig("heatmap.png")
    acc = corrects.double() / len(dataloaders_dict["test"].dataset)
    print('Test Accuracy: {:.4f}'.format(acc))
    return model, acc.cpu().numpy()

def plot_results(train_acc, valid_acc, test_acc, path):
    f,ax = plt.subplots()
    ax.plot(train_acc)
    ax.plot(valid_acc)
    ax.set_ylim(0,1)
    ax.set_title('test: {}'.format(test_acc))
    plt.savefig(path)
    plt.show()

def plot_loss_accuracy(train_accuracy, valid_accuracy, test_accuracy, condition):
    trac,trls,vrac,vrls = [], [], [], []
    for acc,los in train_accuracy:
        trac.append(acc)
        trls.append(los)
    for acc,los in valid_accuracy:
        vrac.append(acc)
        vrls.append(los)

    path_to_image = './results/{}_ACresults.png'.format(condition)
    plot_results(trac, vrac, test_accuracy, path_to_image)
    #send_image(path_to_img=path_to_image, message='Accuracy results')

    path_to_image = './results/{}_LSresults.png'.format(condition)
    plot_results(trls, vrls, test_accuracy, path_to_image)
    #send_image(path_to_img=path_to_image, message='Loss results')


def train_m2dcnn(dataset_path, condition,nb_classes = 619,  batch_size = 128, num_epochs = 300, weights=None):
    seed_everything()

    # DataLoader
    train_dataset = lpp_Dataset(dataset_path[0], language="EN")
    valid_dataset = lpp_Dataset(dataset_path[1], language="EN")
    test_dataset = lpp_Dataset(dataset_path[2], language="EN")

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader, "test": test_dataloader}

    model = M2DCNN(numClass=nb_classes, numFeatues=30880, DIMX=74, DIMY=90, DIMZ=73)
    if weights != None:
        model.load_state_dict(torch.load(weights))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(),lr=0.001,betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    model, train_accuracy, valid_accuracy = train_model_m2dcnn(model, dataloaders_dict, criterion,
                                                        optimizer, scheduler, num_epochs = num_epochs)
    model, test_accuracy = test_model(model, dataloaders_dict, nb_classes)

    plot_loss_accuracy(train_accuracy, valid_accuracy, test_accuracy, condition)
    torch.save(model.state_dict(), './results/{}_weights.pth'.format(condition))
    scipy.io.savemat('./results/{}_results.pth'.format(condition),
            {
                'train_accuracy':train_accuracy,
                'valid_accuracy':valid_accuracy,
                'test_accuracy':test_accuracy,
            }
        )
    
    return test_accuracy

def train_3dcnn(dataset_path, condition, batch_size = 128, num_epochs = 300, nb_classes=32):
    seed_everything()

    # DataLoader
    train_dataset = lpp_Dataset(dataset_path[0])
    valid_dataset = lpp_Dataset(dataset_path[1])
    test_dataset = lpp_Dataset(dataset_path[2])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader, "test": test_dataloader}

    model = CNN3D(n_classes=nb_classes)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(),lr=0.001,betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    model, train_accuracy, valid_accuracy = train_model_m2dcnn(model, dataloaders_dict, criterion,
                                                        optimizer, scheduler, is3D=True, num_epochs = num_epochs)
    print(datetime.datetime.now())
    model, test_accuracy = test_model(model, dataloaders_dict)

    plot_loss_accuracy(train_accuracy, valid_accuracy, test_accuracy, condition)
    torch.save(model.state_dict(), './results/{}_weights.pth'.format(condition))
    scipy.io.savemat('./results/{}_results.pth'.format(condition),
            {
                'train_accuracy':train_accuracy,
                'valid_accuracy':valid_accuracy,
                'test_accuracy':test_accuracy,
            }
        )
    
    return test_accuracy.cpu().numpy()

def calc_baseline(lbl = "data/Test/", language = "EN"):
    prob = 0
    labels = []
    for run in os.listdir(lbl):
        path = lbl +"/"+ run + "/" + language
        multiplier = len(os.listdir(path))-1
        if multiplier>0:
            for _ in range(int(multiplier/len(np.loadtxt(path+"/labels.txt")))):
                labels.append(np.loadtxt(path+"/labels.txt"))
    labels = np.concatenate(labels).ravel()
    total = labels.size
    label_c = Counter(labels)
    for _, v in label_c.items():
        prob += (v/total)**2
    return prob

def write_result(acc, top3, config):
    with open("results/result_log.txt", "a") as f:
        string = f"\n\n{datetime.datetime.now()}\nconfig: {config}\naccuracy : {acc}\nBaseline : {calc_baseline()}\ntop3 accuracy : {top3}"
        f.write(string)
        f.close()

def test(model, config:str, weights_file,nb_classes, dataloader):
    model.load_state_dict(torch.load(weights_file))
    model.eval()
    model.to(device)
    corrects = 0
    top3_corrects = 0
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for inputs, labels in tqdm(dataloader):
       
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, top3preds = torch.topk(outputs, 3)
            corrects += torch.sum(preds == labels.data)
            for g in labels:
                if g in top3preds:
                    top3_corrects += 1
                    continue
            #top3_corrects += torch.eq(labels, top3preds).any(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
    plt.figure()
    sns.heatmap(confusion_matrix/torch.sum(confusion_matrix)*10)
    plt.savefig(f"{config}_heatmap.png")
    acc = corrects.double() / len(dataloader.dataset)
    t3acc = top3_corrects / len(dataloader.dataset)
    print('Test Accuracy: {:.4f}'.format(acc))
    write_result(acc, t3acc, config)
    return model, acc


def train(binary, batch_size, num_epochs, config="config1_EN", model = "2d", weights = None):
    if binary:
        nb_classes = 2
    else:
        with codecs.open("label_dict.txt", "r", encoding="utf-8") as f:
            nb_classes = len(f.readlines())
    path = ["data/Train", "data/Val", "data/Test"]
    t0 = datetime.datetime.now()
    print(t0)
    if model == "2d":
        train_m2dcnn(path, config, nb_classes=nb_classes, batch_size=batch_size, num_epochs=num_epochs, weights=weights)
    else:
        train_3dcnn(path, config, num_epochs=num_epochs, batch_size=batch_size, nb_classes=nb_classes)
    print(datetime.datetime.now()-t0)
if __name__ == "__main__":
    test_dataloader = DataLoader(lpp_Dataset("data/Test/"), batch_size=50, shuffle=False)
    test(M2DCNN(numClass=16, numFeatues=30880, DIMX=74, DIMY=90, DIMZ=73), "config1_EN_top15Nouns","results/config1_EN_top15Noun_weights.pth", 16, test_dataloader)

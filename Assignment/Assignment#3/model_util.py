import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

def train_loop(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_func: nn.CrossEntropyLoss, 
    optimizer: torch.optim, 
    device: str,
    transform: transforms = None,
    google: bool = False,
    hide_process: bool = False):
    """Train model using device."""
    model.train()
    loss_tmp = 0
    acc_tmp = 0
    for x, y in tqdm(data_loader, disable=hide_process):
        x, y = Variable(x).to(device), Variable(y).to(device)
        if transform:
            x = transform(x)
        optimizer.zero_grad()
        yhat = model(x)
        if google:
            loss = loss_func(yhat[0], y) + 0.3*loss_func(yhat[1], y) + 0.3*loss_func(yhat[2], y)
        else:
            loss = loss_func(yhat, y)
        loss.backward()
        optimizer.step()
        loss_tmp += loss.item()     
        if google:
            yhat = yhat[0]
        acc_tmp += (yhat.argmax(1) == y).type(torch.float).sum().item()
        
    loss_tmp /= len(data_loader)
    acc_tmp /= len(data_loader.dataset)
    return acc_tmp, loss_tmp


def evaluate_loop(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_func: nn, 
    device: str,
    hide_process: bool = False):
    """Evaluate Model using given data, and returns accuracy and loss."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader, disable=hide_process):
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss += loss_func(yhat, y).item()
            pred = yhat.softmax(dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    
    loss /= len(data_loader)
    acc = correct / len(data_loader.dataset)
    return acc, loss

def train_eval(
    model: nn.Module, 
    train_loader: DataLoader, 
    valid_loader: DataLoader,
    loss_func: nn.CrossEntropyLoss, 
    optimizer: torch.optim, 
    epochs: int, 
    device: str,
    save_best: bool = True,
    model_name: str = 'best.pth',
    model_path: str = './model/',
    hide_process: bool = False) -> list:
    """Train model epochs times using device, and evaluate it.
    Finally It saves the best accuracy for valid dataset."""
    best_acc = 0.0
    best_at = 0
    hist = []
    # Train
    for epoch in range(1, epochs+1):
        model.train()
        print(f"------------- Epoch {epoch} -------------")
        loss_tmp = 0
        acc_tmp = 0
        for x, y in tqdm(train_loader, disable=hide_process):
            x, y = Variable(x).to(device), Variable(y).to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_func(yhat, y)
            loss.backward()
            optimizer.step()
            loss_tmp += loss.item()     
            acc_tmp += (yhat.argmax(1) == y).type(torch.float).sum().item()
        
        loss_tmp /= len(train_loader)
        acc_tmp /= len(train_loader.dataset)
        hist.append([acc_tmp, loss_tmp])

        # Valid
        model.eval()
        loss_val = 0
        correct_val = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader, disable=hide_process):
                x, y = Variable(x).to(device), Variable(y).to(device)
                yhat = model(x)
                loss_val += loss_func(yhat, y).item()
                pred = yhat.softmax(dim=1).argmax(dim=1, keepdim=True)
                #pred = yhat.argmax(dim=1, keepdim=True)
                correct_val += pred.eq(y.view_as(pred)).sum().item()
        
        loss_val /= len(valid_loader)
        acc_val = correct_val / len(valid_loader.dataset)
        print(f"Train Accuracy: {(100*acc_tmp):>0.1f}%\t\t Train Avg loss: {loss_tmp:>8f}")
        print(f"Valid Accuracy: {(100*acc_val):>0.1f}%\t\t Valid Avg loss: {loss_val:>8f} \n")
        if save_best:
            if acc_val > best_acc:
                best_acc = acc_val
                best_at = epoch
                torch.save(model.state_dict(), model_path+model_name)

    print(f"\nBest at Epochs {best_at} ({100*best_acc:>.1f})")
    return hist


def test_predict(
    model: nn.Module,
    test_loader: DataLoader,
    path: str,
    file_name: str,
    device: str,
    hide_process: bool = False) -> None:
    """Predict test data, and save the results to csv"""
    df = []
    model.eval()
    file = 0
    with torch.no_grad():
        for x in tqdm(test_loader, disable=hide_process):
            x = x.to(device)
            yhat = model(x)
            pred = yhat.softmax(dim=1).argmax(dim=1, keepdim=True)
            df.append([str(file).zfill(3), pred.item()])
            file += 1

    df = pd.DataFrame(df)
    df.columns = ['file_name', 'label']
    path = path.strip('/') + '/' + file_name
    df.to_csv(path, index=False)
    print(f"Model Prediction was saved at {path}")
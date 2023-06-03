from models import Baseline
from datasets import RoIBOLD
import config

import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm, trange
import numpy as np

def main():
    dataset = RoIBOLD(data_csvn='OASIS3_convert_vs_nonconvert.csv')
    train_len = int(config.TRAIN_RATIO*len(dataset))
    trainset, valset = random_split(dataset, [train_len, len(dataset) - train_len], torch.Generator().manual_seed(2345))
    train_class_hist = np.histogram(dataset.labels[torch.LongTensor(trainset.indices)], bins=len(dataset.class_dict))[0]
    print(f"train label histogram: {train_class_hist}, label class: {dataset.class_dict}")
    train_class_hist = np.histogram(dataset.labels[torch.LongTensor(valset.indices)], bins=len(dataset.class_dict))[0]
    print(f"val label histogram: {train_class_hist}, val class: {dataset.class_dict}")
    print("tain length", train_len, "validate length", len(dataset) - train_len)
    # exit()
    train_batch_size = config.BATCH_SIZE
    val_batch_size = config.BATCH_SIZE
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=16, collate_fn=dataset.collate_fn)

    model = Baseline().to(config.DEVICE)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    # Define your optimizer
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(config.num_epochs*0.6), gamma=0.1)

    best_acc = 0 
    for epoch in trange(config.num_epochs):
        train_loss, train_acc, sub_train_acc, train_acc2 = train(model, train_loader, criterion, optimizer, scheduler, epoch)
        val_loss, val_acc, sub_val_acc, val_acc2 = validate(model, val_loader, criterion, epoch)
        print(f"Epoch {epoch+1:04d} | Train Loss: {train_loss:.5f} | Sample Train Acc: {train_acc:.5f} | Val Loss: {val_loss:.5f} | Sample Val Acc: {val_acc:.5f}")
        print(f"             Subject Train Acc: {sub_train_acc:.5f} | Train_acc_balance: {train_acc2:.5f} | Subject Val Acc: {sub_val_acc:.5f} | Val_acc_balance: {val_acc2:.5f}")
        torch.save(model.state_dict(), "%s/lastest.pth" % (config.SAVE_DIR))
        if best_acc <= sub_val_acc: torch.save(model.state_dict(), "%s/best.pth" % (config.SAVE_DIR))
        scheduler.step()



def train(model, dataloader, criterion, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    y_true = []
    y_pred = []
    sub_true = []
    sub_pred = []
    sub_ids = []
    sub_scores = []
    for batch_idx, (data, target, sid) in enumerate(dataloader):
        # data = [d.to(config.DEVICE) for d in data]
        target = target.to(config.DEVICE)
        sid = sid.to(config.DEVICE)
        optimizer.zero_grad()
        output = torch.cat([model(d.unsqueeze(0).to(config.DEVICE)) for d in data])
        score, pred = torch.max(output, dim=1)
        train_correct += torch.sum(pred == target)
        y_true += target.tolist()
        y_pred += pred.tolist()
        sub_id, strue, spred, sub_score, indecies = get_subject_acc(sid, target, pred, score)
        sub_true.append(strue)
        sub_pred.append(spred)
        sub_ids.append(sub_id)
        sub_scores.append(sub_score)
        loss = criterion(output, target) # loss of all time points 
        train_loss += loss.item() * len(data)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            # print('Train/Loss', loss.item(), step)
            # print('Train/Learning Rate', scheduler.get_last_lr()[0], step)
    train_loss /= len(dataloader.dataset)
    train_acc = train_correct.float() / len(dataloader.dataset)
    sub_id, strue, spred, sub_score, _ = get_subject_acc(torch.cat(sub_ids), torch.cat(sub_true), torch.cat(sub_pred), torch.cat(sub_scores))
    # assert len(sub_id) == len(dataloader.dataset.dataset.subject_names)
    sub_acc = torch.sum(spred == strue).float() / len(spred)
    sub_acc0 = torch.sum(spred[strue==0] == strue[strue==0]).float() / len(spred[strue==0])
    sub_acc1 = torch.sum(spred[strue==1] == strue[strue==1]).float() / len(spred[strue==1])

    return train_loss, train_acc, sub_acc, (sub_acc0+sub_acc1)/2


def validate(model, dataloader, criterion, epoch):
    model.eval()
    val_loss = 0
    val_correct = 0
    y_true = []
    y_pred = []
    sub_true = []
    sub_pred = []
    sub_ids = []
    sub_scores = []
    with torch.no_grad():
        for data, target, sid in dataloader:
            # data = data.to(config.DEVICE)
            target = target.to(config.DEVICE)
            sid = sid.to(config.DEVICE)
            # output = model(data)
            output = torch.cat([model(d.unsqueeze(0).to(config.DEVICE)) for d in data])
            score, pred = torch.max(output, dim=1)
            val_correct += torch.sum(pred == target)
            y_true += target.tolist()
            y_pred += pred.tolist()
            sub_id, strue, spred, sub_score, indecies = get_subject_acc(sid, target, pred, score)
            sub_true.append(strue)
            sub_pred.append(spred)
            sub_ids.append(sub_id)
            sub_scores.append(sub_score)
            loss = criterion(output, target) # loss of all time points 
            val_loss += loss.item() * len(data)
        val_loss /= len(dataloader.dataset)
        val_acc = val_correct.float() / len(dataloader.dataset)
        sub_id, strue, spred, sub_score, _ = get_subject_acc(torch.cat(sub_ids), torch.cat(sub_true), torch.cat(sub_pred), torch.cat(sub_scores))
        # assert len(sub_id) == len(dataloader.dataset.dataset.subject_names)
        sub_acc = torch.sum(spred == strue).float() / len(spred)
        sub_acc0 = torch.sum(spred[strue==0] == strue[strue==0]).float() / len(spred[strue==0])
        sub_acc1 = torch.sum(spred[strue==1] == strue[strue==1]).float() / len(spred[strue==1])

    return val_loss, val_acc, sub_acc, (sub_acc0+sub_acc1)/2

def get_subject_acc(subject_ids, targets, preds, scores):
    strue = []
    spred = []
    sids = []
    sscore = []
    indecies = []
    for si in subject_ids.unique():
        tgt = targets[subject_ids==si]
        pred = preds[subject_ids==si]
        score = scores[subject_ids==si]
        assert len(tgt.unique()) == 1, f"subject {si} has wrong label: {tgt.unique()}"
        strue.append(tgt[0])
        spred.append(pred[score.argmax()])
        sscore.append(score.max())
        # spred.append(1 if score[pred==1].mean() > score[pred==0].mean() else 0)
        # sscore.append(score.mean())
        sids.append(si)
        indecies.append(torch.where(subject_ids==si)[0][score.argsort()[:1]])
    return torch.stack(sids), torch.stack(strue), torch.stack(spred), torch.stack(sscore), torch.cat(indecies)

if __name__ == '__main__':
    main()
from models import Baseline
from datasets import RoIBOLD
import config

import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm, trange


def main():
    dataset = RoIBOLD(data_csvn='OASIS3_convert_vs_nonconvert.csv')
    train_len = int(config.TRAIN_RATIO*len(dataset))
    trainset, valset = random_split(dataset, [train_len, len(dataset) - train_len], torch.Generator().manual_seed(2345))

    train_batch_size = config.BATCH_SIZE
    val_batch_size = config.BATCH_SIZE
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)#, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=16)#, collate_fn=val_data.collate_fn)

    model = Baseline().to(config.DEVICE)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    # Define your optimizer
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(config.num_epochs*0.6), gamma=0.1)

    best_acc = 0 
    for epoch in trange(config.num_epochs):
        train_loss, train_acc, sub_train_acc = train(model, train_loader, criterion, optimizer, scheduler, epoch)
        val_loss, val_acc, sub_val_acc = validate(model, val_loader, criterion, epoch)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Subject Train Acc: {sub_train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f} | Subject Val Acc: {sub_val_acc:.3f}")
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
        data = data.to(config.DEVICE)
        target = target.to(config.DEVICE)
        sid = sid.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(data)
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
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            print('Train/Loss', loss.item(), step)
            print('Train/Learning Rate', scheduler.get_last_lr()[0], step)
    train_loss /= len(dataloader.dataset)
    train_acc = train_correct.float() / len(dataloader.dataset)
    sub_id, strue, spred, sub_score, _ = get_subject_acc(torch.cat(sub_ids), torch.cat(sub_true), torch.cat(sub_pred), torch.cat(sub_scores))
    assert len(sub_id) == len(dataloader.dataset.subject_dict)
    sub_acc = torch.sum(spred == strue).float() / len(spred)

    return train_loss, train_acc, sub_acc


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
            data = data.to(config.DEVICE)
            target = target.to(config.DEVICE)
            sid = sid.to(config.DEVICE)
            output = model(data)
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
            val_loss += loss.item() * data.size(0)
        val_loss /= len(dataloader.dataset)
        val_acc = val_correct.float() / len(dataloader.dataset)
        sub_id, strue, spred, sub_score, _ = get_subject_acc(torch.cat(sub_ids), torch.cat(sub_true), torch.cat(sub_pred), torch.cat(sub_scores))
        assert len(sub_id) == len(dataloader.dataset.subject_dict)
        sub_acc = torch.sum(spred == strue).float() / len(spred)

    return val_loss, val_acc, sub_acc

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
        indecies.append(torch.where(subject_ids==si)[0][score.argsort()[0]])
    return torch.stack(sids), torch.stack(strue), torch.stack(spred), torch.stack(sscore), torch.cat(indecies)

if __name__ == '__main__':
    main()
from models import OursSelfCorr
from datasets import RoIBOLD, RoIBOLDCorrCoefWin
import config
from data_proc import bold_signal_to_trends, bold_signal_threshold

import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from tqdm import tqdm, trange
import numpy as np

def main():
    roi_start, roi_end = config.DATA['roi_start'], config.DATA['roi_end']
    data_csvn = config.DATA['data_csvn'] 
    dataset = RoIBOLD(data_csvn=data_csvn, roi_start=roi_start, roi_end=roi_end,
                        # roi_num=90,
                        # preproc=bold_signal_to_trends,
                        # preproc=bold_signal_threshold,
                    )

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
    class_weights = torch.from_numpy(sum(train_class_hist)/train_class_hist).float()#.to(config.DEVICE)
    # sampler = WeightedRandomSampler([class_weights[c] for c in dataset.labels[torch.LongTensor(trainset.indices)]], train_batch_size)
    # train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=False, sampler=sampler, num_workers=16, collate_fn=dataset.collate_fn)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=16, collate_fn=dataset.collate_fn)

    model = OursSelfCorr(dataset.roi_num).to(config.DEVICE)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    # Define your optimizer
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
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
        output = []
        corr_loss = []
        for d in data:
            out, closs = model(d.unsqueeze(0).to(config.DEVICE))
            output.append(out)
            corr_loss.append(closs)
        output = torch.cat(output)
        if corr_loss[0] is not None:
            corr_loss = torch.stack(corr_loss)
        score, pred = torch.max(output, dim=1)
        train_correct += torch.sum(pred == target)
        y_true += target.tolist()
        y_pred += pred.tolist()
        sub_id, strue, spred, sub_score, indecies = get_subject_acc(sid, target, pred, score)
        sub_true.append(strue)
        sub_pred.append(spred)
        sub_ids.append(sub_id)
        sub_scores.append(sub_score)
        if corr_loss[0] is not None:
            loss = criterion(output, target) + corr_loss.mean()
        else:
            loss = criterion(output, target) 
        train_loss += loss.item() * len(data)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            print('Train/Loss', loss.item(), corr_loss.mean().item() if corr_loss[0] is not None else 0, step)
            print('Train/Learning Rate', scheduler.get_last_lr()[0], step)
    train_loss /= len(dataloader.dataset)
    train_acc = train_correct.float() / len(dataloader.dataset)
    sub_id, strue, spred, sub_score, _ = get_subject_acc(torch.cat(sub_ids), torch.cat(sub_true), torch.cat(sub_pred), torch.cat(sub_scores))
    # assert len(sub_id) == len(dataloader.dataset.dataset.subject_names)
    sub_acc = torch.sum(spred == strue).float() / len(spred)
    balance_acc = 0
    for i in range(config.BASELINE_MODEL['nclass']):
        balance_acc += (torch.sum(spred[strue==i] == strue[strue==i]).float() / len(spred[strue==i])) / config.BASELINE_MODEL['nclass']

    return train_loss, train_acc, sub_acc, balance_acc


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
        for batch_idx, (data, target, sid) in enumerate(dataloader):
            # data = data.to(config.DEVICE)
            target = target.to(config.DEVICE)
            sid = sid.to(config.DEVICE)
            # output = model(data)
            output = []
            corr_loss = []
            for d in data:
                out, closs = model(d.unsqueeze(0).to(config.DEVICE))
                output.append(out)
                corr_loss.append(closs)    
            if corr_loss[0] is not None:
                corr_loss = torch.stack(corr_loss) 
            output = torch.cat(output) 
            score, pred = torch.max(output, dim=1)
            val_correct += torch.sum(pred == target)
            y_true += target.tolist()
            y_pred += pred.tolist()
            sub_id, strue, spred, sub_score, indecies = get_subject_acc(sid, target, pred, score)
            sub_true.append(strue)
            sub_pred.append(spred)
            sub_ids.append(sub_id)
            sub_scores.append(sub_score)
            if corr_loss[0] is not None:
                loss = criterion(output, target) + corr_loss.mean()
            else:
                loss = criterion(output, target) 
            val_loss += loss.item() * len(data)
            if batch_idx % 10 == 0:
                step = epoch * len(dataloader) + batch_idx
                print('Val/Loss', loss.item(), corr_loss.mean().item() if corr_loss[0] is not None else 0, step)
        val_loss /= len(dataloader.dataset)
        val_acc = val_correct.float() / len(dataloader.dataset)
        sub_id, strue, spred, sub_score, _ = get_subject_acc(torch.cat(sub_ids), torch.cat(sub_true), torch.cat(sub_pred), torch.cat(sub_scores))
        # assert len(sub_id) == len(dataloader.dataset.dataset.subject_names)
        sub_acc = torch.sum(spred == strue).float() / len(spred)
        balance_acc = 0
        for i in range(config.BASELINE_MODEL['nclass']):
            balance_acc += (torch.sum(spred[strue==i] == strue[strue==i]).float() / len(spred[strue==i])) / config.BASELINE_MODEL['nclass']

    return val_loss, val_acc, sub_acc, balance_acc

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
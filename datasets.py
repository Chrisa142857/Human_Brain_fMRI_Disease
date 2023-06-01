from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch, csv


def load_label(label_csvn='../data/OASIS3/clinical_data/ADRC_clinical.csv') -> {}:
    '''
     Cognitively normal: CN
     Others: AD
    '''
    subject_label = {}
    with open(label_csvn, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    data = list(spamreader)
    
    with open(label_csvn, 'r') as f:
        lines = f.read().split('\n')[1:-1]
    for line in lines:
        items = line.split(',')
        subject_n = items[1].replace('_', '')
        if items[-1] not in class_map: continue
        subject_label[subject_n] = class_map[items[-1]]
    return subject_label

class RoIBOLDCorrCoefMat(Dataset):

    def __init__(self, flist, seq_len, ) -> None:
        from trainval import step_size as STEP_SIZE
        step_size = STEP_SIZE
        self.flist = []
        self.labels = []
        self.subjects = []
        subject_label = load_label()
        self.seq_len = seq_len
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(subject_label.values())))}
        self.subject_dict = {}
        self.data = []
        for fpath in tqdm(flist, desc="init dataset"):
            subject_n = fpath.split('/')[-1][4:-8]
            if subject_n not in subject_label: continue
            data = np.loadtxt(fpath)[:, :90] # Time x RoI
            # data = (data - data.min()) / (data.max() - data.min())
            data = [
                np.corrcoef(data[st:st+seq_len].T) for st in range(0, len(data), step_size)
            ]
            self.data.extend(data)
            self.labels.extend([self.class_dict[subject_label[subject_n]] for _ in range(len(data))])
            if subject_n not in self.subject_dict: self.subject_dict[subject_n] = len(self.subject_dict)
            self.subjects.extend([self.subject_dict[subject_n] for _ in range(len(data))])

        self.data = torch.from_numpy(np.stack(self.data).astype(np.float32))

    def __getitem__(self, idx):
        data = self.data[idx]
        assert not np.isnan(data).any(), "%d data has nan" % idx
        return data, self.labels[idx], self.subjects[idx]

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        data = []
        labels = []
        subs = []
        for d, label, sub in batch:
            data.append(torch.from_numpy(d))
            labels.append(torch.LongTensor([label]))
            subs.append(torch.LongTensor([sub]))
            # d = torch.from_numpy(d)
            # if d.shape[0] >= self.seq_len:
            #     st = torch.randint(low=0, high=d.shape[0]-self.seq_len, size=(1,))
            #     data.append(d[st:st+self.seq_len])
            # else:
            #     data.append(torch.cat([d, torch.zeros(self.seq_len-d.shape[0], d.shape[1], dtype=d.dtype)], dim=0))
        data = torch.stack(data)
        return data, torch.cat(labels), torch.cat(subs)

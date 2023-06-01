from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch, csv, os
STEP_SIZE = 30
WIN_SIZE = 100

class RoIBOLDCorrCoefMat(Dataset):
    def __init__(self, flist, label_csvn=None, preproc=np.corrcoef, roi_num=191) -> None:
        step_size = STEP_SIZE
        seq_len = WIN_SIZE
        with open(label_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = flist
        self.labels = []
        self.subject_id_list = []
        self.seq_len = seq_len
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_id_dict = {}
        self.data = []
        for fpath in tqdm(flist, desc="init dataset"):
            subject_n = fpath.split('/')[-1].split('_')[0]
            label_key = fpath.split('/')[-1][:-4]
            if label_key not in self.label_dict: continue
            data = np.loadtxt(fpath)[:, :roi_num] # Time x RoI
            if preproc is not None:
                data = [
                    preproc(data[st:st+seq_len].T) for st in range(0, len(data), step_size)
                ]
                self.data.extend(data)
                self.labels.extend([self.class_dict[self.label_dict[label_key]] for _ in range(len(data))])
                if subject_n not in self.subject_id_dict: self.subject_id_dict[subject_n] = len(self.subject_id_dict)
                self.subject_id_list.extend([self.subject_id_dict[subject_n] for _ in range(len(data))])
            else:
                self.data.append(data)
                self.labels.append(self.class_dict[self.label_dict[label_key]])
                if subject_n not in self.subject_id_dict: self.subject_id_dict[subject_n] = len(self.subject_id_dict)
                self.subject_id_list.append(self.subject_id_dict[subject_n])

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


if __name__ == '__main__':
    r = '../data/OASIS3/fMRI_processed/RoI_BOLD/a2009s_ReadyForTrain'
    label_csvn='../data/OASIS3/fMRI_label.csv'
    flist = os.listdir(r)
    flist = [os.path.join(r, f) for f in flist]
    RoIBOLDCorrCoefMat(flist, label_csvn, preproc=None)
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch, csv, os
STEP_SIZE = 30
WIN_SIZE = 100

class RoIBOLD(Dataset):
    def __init__(self, data_csvn=None, roi_num=191) -> None:
        # step_size = STEP_SIZE
        # seq_len = WIN_SIZE
        with open(data_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = [
            l.split(',')[3].split('@')
        for l in lines]
        self.labels = []
        # self.seq_len = seq_len
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_names = []
        self.subject_id_list = []
        self.data = []
        sid = 0
        for fpaths in tqdm(self.flist, desc="init dataset"):
            data = []
            for fpath in fpaths:
                data.append(np.loadtxt(fpath)[:, :roi_num]) # Time x RoI
            subject_n = fpath.split('/')[-1].split('_')[0]
            self.data.append(torch.from_numpy(np.concatenate(data).astype(np.float32)))
            self.labels.append(self.class_dict[self.label_dict[subject_n]])
            self.subject_id_list.append(sid)
            if subject_n not in self.subject_names: 
                self.subject_names.append(subject_n)
                sid += 1

        # self.data = torch.from_numpy(np.stack(self.data).astype(np.float32))
        self.labels = torch.LongTensor(self.labels)
        self.subject_id_list = torch.LongTensor(self.subject_id_list)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_id_list[idx]

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        data = []
        labels = []
        subs = []
        for d, label, sub in batch:
            data.append(d)
            labels.append(torch.LongTensor([label]))
            subs.append(torch.LongTensor([sub]))
            # d = torch.from_numpy(d)
            # if d.shape[0] >= self.seq_len:
            #     st = torch.randint(low=0, high=d.shape[0]-self.seq_len, size=(1,))
            #     data.append(d[st:st+self.seq_len])
            # else:
            #     data.append(torch.cat([d, torch.zeros(self.seq_len-d.shape[0], d.shape[1], dtype=d.dtype)], dim=0))
        # data = torch.stack(data)
        return data, torch.cat(labels), torch.cat(subs)


if __name__ == '__main__':
    # r = '../data/OASIS3/fMRI_processed/RoI_BOLD/a2009s_ReadyForTrain'
    # label_csvn='../data/OASIS3/fMRI_label.csv'
    # flist = os.listdir(r)
    # flist = [os.path.join(r, f) for f in flist]
    # RoIBOLDCorrCoefMat(flist, label_csvn, preproc=None)

    RoIBOLD(data_csvn='OASIS3_convert_vs_nonconvert.csv')
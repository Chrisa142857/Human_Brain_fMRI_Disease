from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch, csv, os
from tqdm import trange
from config import STEP_SIZE, WIN_SIZE

class RoIBOLD(Dataset):
    def __init__(self, data_csvn=None, roi_num=191, preproc=None) -> None:
        # step_size = STEP_SIZE
        # seq_len = WIN_SIZE
        self.roi_num = None
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
            data = torch.from_numpy(np.concatenate(data).astype(np.float32))
            if preproc:
                data = preproc(data)
            self.data.append(data)
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


class RoIBOLDCorrCoef(Dataset): ## Each data is one CC mat of a subject (1x150x150)
    def __init__(self, data_csvn=None, roi_start=41, roi_end=191, preproc=None) -> None:
        step_size = STEP_SIZE
        seq_len = WIN_SIZE
        self.roi_num = roi_end - roi_start
        with open(data_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = [
            l.split(',')[3].split('@')
        for l in lines]
        self.labels = []
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_names = []
        self.subject_id_list = []
        self.data = []
        sid = 0
        for fpaths in tqdm(self.flist, desc="init dataset"):
            data = []
            for fpath in fpaths:
                data.append(np.loadtxt(fpath)[:, roi_start:roi_end]) # Time x RoI
            subject_n = fpath.split('/')[-1].split('_')[0]
            data = torch.from_numpy(np.concatenate(data).astype(np.float32))
            if preproc:
                data = preproc(data)
            data = torch.stack([corrcoef(data[st:st+seq_len].T) for st in range(0, len(data), step_size)])
            for d in data:
                self.data.append(d)
                self.labels.append(self.class_dict[self.label_dict[subject_n]])
                self.subject_id_list.append(sid)
            if subject_n not in self.subject_names: 
                self.subject_names.append(subject_n)
                sid += 1

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
        return torch.stack(data), torch.cat(labels), torch.cat(subs)


class RoIBOLDCorrCoefWin(Dataset): ## Each data is the sequence of one subject (Tx150x150)
    def __init__(self, data_csvn=None, roi_start=41, roi_end=191, preproc=None) -> None:
        step_size = STEP_SIZE
        seq_len = WIN_SIZE
        self.roi_num = roi_end - roi_start
        with open(data_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = [
            l.split(',')[3].split('@')
        for l in lines]
        self.labels = []
        self.seq_len = seq_len
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_names = []
        self.subject_id_list = []
        self.data = []
        sid = 0
        for fpaths in tqdm(self.flist, desc="init dataset"):
            data = []
            for fpath in fpaths:
                data.append(np.loadtxt(fpath)[:, roi_start:roi_end]) # Time x RoI
            subject_n = fpath.split('/')[-1].split('_')[0]
            data = torch.from_numpy(np.concatenate(data).astype(np.float32))
            if preproc:
                data = preproc(data)
            self.data.append([data[st:st+seq_len] for st in range(0, len(data), step_size)])
            self.labels.append(self.class_dict[self.label_dict[subject_n]])
            self.subject_id_list.append(sid)
            if subject_n not in self.subject_names: 
                self.subject_names.append(subject_n)
                sid += 1

        for i in trange(len(self.data), desc='Get corr coef SPD matrix'):
            self.corrcoef_spd(i)
        self.labels = torch.LongTensor(self.labels)
        self.subject_id_list = torch.LongTensor(self.subject_id_list)

    def corrcoef_spd(self, idx):
        data = torch.stack([corrcoef(d.T) for d in self.data[idx]])
        data = data[~data.isnan().any(1).any(1)]
        # torch.nn.init.orthogonal_(data)
        # data = data + 1e-10*torch.stack([torch.eye(data.shape[-1]) for _ in range(len(data))], -1).permute(2,0,1)
        self.data[idx] = data

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
        return data, torch.cat(labels), torch.cat(subs)


def corrcoef(X):
    avg = torch.mean(X, dim=-1)
    X = X - avg[..., None]
    X_T = X.swapaxes(-2, -1)
    c = torch.matmul(X, X_T)
    d = torch.diagonal(c, 0, -2, -1)
    stddev = torch.sqrt(d.clone().detach())
    c = c / stddev[..., None]
    c = c / stddev[..., None, :]
    c = torch.clip(c, -1, 1, out=c)
    c[c.isnan()] = 0
    # c = nearestPD(c)
    return c
    # # ChatGPT answers:
    # # 计算相关性系数矩阵
    # # 输入:
    # #   X: 输入数据，形状为 (num_samples, num_variables)
    # # 输出:
    # #   corr_matrix: 相关性系数矩阵，形状为 (num_variables, num_variables)
    # # 计算均值
    # mean_X = torch.mean(X, dim=0)
    # # 计算标准差
    # std_X = torch.std(X, dim=0)
    # # 归一化数据
    # X_normalized = (X - mean_X) / std_X
    # # 计算协方差矩阵
    # cov_matrix = torch.matmul(X_normalized.T, X_normalized) / X.shape[0]
    # # 计算相关性系数矩阵
    # corr_matrix = cov_matrix / torch.sqrt(torch.diag(cov_matrix))
    # corr_matrix = nearestPD(corr_matrix)
    # return corr_matrix

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = torch.linalg.svd(B)
    H = V.T @ (torch.diag(s) @ V)

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3
    
    spacing = np.spacing(torch.linalg.norm(A).numpy())
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.real(torch.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.linalg.cholesky(B)
        _ = torch.linalg.svd(B)
        return True
    except torch.linalg.LinAlgError:
        return False


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from models import BaselineSPD
    # input_size = 150
    # out_size = 18
    input_size = 90
    out_size = 11
    # net = BaselineSPD(input_size)
    # net.load_state_dict(torch.load('work_dir/ADNI/best.pth'))
    # net = net.spdnet
    # net.eval()
    # row_idx, col_idx = np.triu_indices(out_size)
    # row_idx = torch.LongTensor(row_idx)
    # col_idx = torch.LongTensor(col_idx)
    dataset = RoIBOLDCorrCoefWin(
        # data_csvn='OASIS3_convert_vs_nonconvert.csv', 
        data_csvn='ADNI_AAL90_5class.csv', roi_start=0, roi_end=90,
    )
    fig, ax = plt.subplots(5,15, figsize=(30,10), layout='tight')#
    # ax = ax.reshape(-1)
    # plt.rcParams['axes.titlesize'] = 5
    class_dict = {v: k for k,v in dataset.class_dict.items()}
    plt_num = [0 for _ in range(5)]
    for di, data in enumerate(dataset):
        label = data[1]
        data = data[0]
        # run SPDnet
        # out = torch.zeros(len(data), out_size, out_size)
        # with torch.no_grad():
        #     data = net(data)
        # out[:, row_idx, col_idx] = data
        # out = out.permute(0, 2, 1)
        # out[:, row_idx, col_idx] = data
        # out = out.permute(0, 2, 1)
        # data = out
        # done SPDnet
        # if di >= 10: continue
        if plt_num[label] >= 15: continue
        ax[label, plt_num[label]].matshow(data[0])
        ax[label, plt_num[label]].set_title(class_dict[label.item()], size=15)
        ax[label, plt_num[label]].axis('off')  
        plt_num[label] += 1
        
    # plt.tight_layout()
    plt.savefig('CCmats_nearestPD_ADNI/0-150_win0.jpg', dpi=600)
    plt.close()
        
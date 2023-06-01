
def class_decode(classn):
    if 'uncertain' in classn or 'Unc' in classn or '0.5' in classn: return None
    if 'AD' in classn and 'Non' not in classn and 'NON' not in classn:
        return 'AD'
    if 'Cognitively normal' in classn or 'No dementia' == classn:
        return 'CN'
    if '' == classn or '.' == classn: return None
    if 'Dementia' in classn or 'demt.' in classn: return 'Dementia'
    if 'DLBD' in classn: return 'DLBD'
    if 'DAT' in classn: return 'DAT'
    if 'Vascular' in classn: return 'Vascular'
    if 'Other' in classn or 'ProAph' in classn: return 'Other'
    return classn


def load_label(label_csvn):
    '''
     Cognitively normal: CN
     Others: AD
    '''
    subject_label = {}
    subject_label_orig = {}
    subject_day = {}
    with open(label_csvn, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        lines = list(spamreader)[1:]
    for line in lines:
        assert line[0] not in subject_label
        sid = line[0].replace('ClinicalData_', '')
        sn = sid.split('_')[0]
        classn = class_decode(line[8])
        if classn is None: continue
        subject_label[sid] = classn
        subject_label_orig[sid] = line[8]
        if sn not in subject_day: subject_day[sn] = []
        subject_day[sn].append(int(sid.split('_')[1][1:]))
    
    return subject_label, subject_day, subject_label_orig

def refine_label(flist, label_csvn, save_csv):
    subject_label, subject_day, subject_label_orig = load_label(label_csvn)
    days = {}
    for f in flist:
        sid = f.split('/')[-1].split('_')[0]
        day = f.split('/')[-1].split('_')[1][1:-4]
        if sid not in days: days[sid] = []
        days[sid].append(int(day))
    day_to_labelday = {}
    for sid in days:
        if sid not in subject_day: continue
        label_days = subject_day[sid]
        fday = days[sid]
        day_to_labelday[sid] = {}
        for d in fday:
            dis = [d - ld if ld <= d else 10000 for ld in label_days]
            ld = label_days[np.argmin(dis)]
            day_to_labelday[sid][d] = ld
    csvhead = 'SUBJECT_ID,LABEL,Original Name\n'
    for fpath in flist:
        subject_n = fpath.split('/')[-1].split('_')[0]
        day = int(fpath.split('/')[-1].split('_')[1][1:-4])
        if subject_n not in subject_day: continue
        label_key = subject_n + '_d%04d' % day_to_labelday[subject_n][day]
        label = subject_label[label_key]
        csvhead += '%s, %s, %s\n' % (fpath.split('/')[-1][:-4], label, subject_label_orig[label_key])
    with open(save_csv, 'w') as f:
        f.write(csvhead)


def rename_bold_data():
    import os, shutil
    import numpy as np

    def merge_names(names):
        out = {}
        day_tag = 's-d'
        for n in names:
            sid = n[4:12]
            day = n[n.index(day_tag)+len(day_tag):].split('_')[0]
            merged_name = sid+'_'+day
            if merged_name not in out: out[merged_name] = []
            out[merged_name].append(n)
        return out
        

    r = '../data/OASIS3/fMRI_processed/RoI_BOLD/a2009s'
    save_r = '../data/OASIS3/fMRI_processed/RoI_BOLD/a2009s_norun'
    flist = os.listdir(r)
    merged = merge_names(flist)
    os.makedirs(save_r, exist_ok=True)

    for save_n in merged.keys():
        out = []
        for i, fn in enumerate(merged[save_n]):
            path = os.path.join(r, fn)
            # shutil.copy(path, os.path.join(save_r, fn+'_run%d.txt'%(i)))
            data = np.loadtxt(path)
            out.append(data)
        
        out = np.concatenate(out)
        np.savetxt(os.path.join(save_r, save_n+'.txt'), out)


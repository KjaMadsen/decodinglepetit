import numpy as np
from torch.utils.data import Dataset
import scipy.stats
import os
import nibabel as nib
import pandas as pd

class lpp_Dataset(Dataset):
    def __init__(self, data_dir, anno_dir, language='EN', region=False):
        self.region = region
        self.subjects = []
        self.label_dict = self.load_label_dict(anno_dir)
        for run in os.listdir(data_dir):
            for dirpath,_,files in os.walk(data_dir+f"/{run}/{language}"):
                for _, scan in enumerate(sorted(files)):
                    if scan.endswith(".nii.gz"):
                        self.subjects.append((run, os.path.join(dirpath, scan)))
                    
        self.scans = []
        self.imgs = []
        for scan in self.subjects:
            run, img = scan
            nii = nib.load(img)
            self.scans.append((run, nii))
        for t in self.scans:
            self.imgs.append(t)
                    

    def __getitem__(self, index):
        run, raw = self.imgs[index]
        if self.region:
            img = self.normalize_data(raw)[self.region]
        else:
            img = self.normalize_data(raw)
        target = self.label_dict[run][index][1]
        return img, target
    
    def __len__(self):
        return len(self.label_dict)
    
    def normalize_data(self, data):
        #need to normalize?
        return np.array(data.dataobj)
    
    def load_label_dict(path):
        file = pd.read_csv(path)
        # adds words that happen in transistion between scans
        # to both scans it appears in
        result = []
        tmp = []
        for idx, row in file.iterrows():
            if True: #row["pos"] == "VERB":
                w = row["lemma"]
                if (row["onset"]-len(result)*2) < 2:
                    tmp.append(w)
                if (row["offset"]-len(result)*2) > 2:
                    result.append((row["section"]-1, tmp, len(result)))
                    if (row["offset"]-len(result)*2) < 2:
                        tmp = [w]
                    else:
                        tmp = []
            else:
                result.append(tmp)
                tmp = []
        label_dict = {k:[]for k in range(9)}
        for run, words, idx in result:
            label_dict[run].append((words, idx))
        return label_dict

    
            
            


class mt_Dataset(Dataset):
    def __init__(self, file_list, label_list, boxcar=False):
        self.boxcar = boxcar
        self.file_list = file_list
        self.label_list = label_list

    def __getitem__(self, index):
        img = self.normalize_data(np.load(self.file_list[index]))
        target = np.argmax(self.label_list[index])
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        # Data auguentaion
        if self.boxcar:
            data = data[:, :, 2:-4, 4:8]
        else:
            data = np.mean(data[:, :, 2:-4, 4:8],axis=3)
        
        data = scipy.stats.zscore(data, axis=None)
        data[~ np.isfinite(data)] = 0

        if self.boxcar:
            return data.transpose(3, 0, 1, 2)
        else:
            return data.transpose(2, 1, 0)
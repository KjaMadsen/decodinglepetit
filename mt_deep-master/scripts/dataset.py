import numpy as np
from torch.utils.data import Dataset
import scipy.stats
import os
import nibabel as nib

class lpp_Dataset(Dataset):
    def __init__(self, data_dir, labels_dict, language='EN', region=False):
        self.region = region
        self.label_dict = labels_dict # = {0: [0,1,2,...], ..., 4 : [3,44,65]}, {run: word embed}
        self.subjects = []
        for dirpath,_,files in os.walk(data_dir+f"/{language}"):
            for run, scan in enumerate(sorted(files)):
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
        target = self.label_dict[run][index]
        return img, target
    
    def __len__(self):
        return len(self.label_dict)
    
    def normalize_data(self, data):
        #need to normalize?
        return np.array(data.dataobj)
    
    
            
            


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
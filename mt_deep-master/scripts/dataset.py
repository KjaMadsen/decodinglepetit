import numpy as np
from torch.utils.data import Dataset
import scipy.stats
import os
import nibabel as nib

class lpp_Dataset(Dataset):
    def __init__(self, data_dir, language='EN', region=False):
        self.region = region
        self.subjects = []
        self.label_dict = {}
        for run in os.listdir(data_dir):
            for dirpath,_,files in os.walk(data_dir+f"/{run}/{language}"):
                for _, file in enumerate(sorted(files)):
                    if file.endswith(".nii.gz"):
                        file_path = os.path.join(dirpath, file)
                        self.subjects.append((run, nib.load(file_path)))
                    elif file.endswith(".txt"):
                        self.label_dict[int(run)] = np.loadtxt(os.path.join(dirpath, file), dtype=int)
                    
        self.scans = []
        self.imgs = dict()
        
        
    def __getitem__(self, index):
        #TODO: decide what to do with scans that
        # does not contain the desired labels
        
        self.cache_data(index)
        run, raw = self.imgs[index]
        if self.region:
            img = self.normalize_data(raw[index])[self.region]
        else:
            img = self.normalize_data(raw[index])
        target = self.label_dict[int(run)][index%282]
        return img, float(target)
    
    def cache_data(self, index):
        run,sub = self.subjects[int(index/282)]
        self.images[index] =  (run, np.array(sub.dataobj).transpose(3, 0, 1, 2)) 
    
    def __len__(self):
        return len(self.label_dict)
    
    #def normalize_data(self, data):
        #need to normalize?
    #   return np.array(data.dataobj)

    def normalize_data(self, data):
        # Data auguentaion
        data = scipy.stats.zscore(data, axis=None)
        data[~ np.isfinite(data)] = 0
        return data
        
    

    
            
            


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
import numpy as np
from torch.utils.data import Dataset
import scipy.stats
import os

class lpp_Dataset(Dataset):
    def __init__(self, data_dir, language='EN', region=False):
        self.region = region
        self.data_dir = data_dir
        self.language = language
        self.label_dict = dict()
        self.run_index = len(data_dir)+1
        self.file_list = []
        for run in os.listdir(data_dir):
            for dirpath,_,files in os.walk(data_dir+f"/{run}/{language}"):
                for _, file in enumerate(sorted(files)):
                    if file.endswith(".txt"):
                        self.label_dict[int(run)] = np.loadtxt(os.path.join(dirpath, file), dtype=int)
                    elif file.endswith(".npy"):
                        self.file_list.append(os.path.join(dirpath, file))
        
        
    def __getitem__(self, index):
        img = self.normalize_data(np.load(self.file_list[index]))
        run = int(self.file_list[index][self.run_index])
        target = self.label_dict[run][index]
        return img, target
    
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
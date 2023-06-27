import numpy as np
from torch.utils.data import Dataset
import scipy.stats
import os

class lpp_Dataset(Dataset):
    def __init__(self, data_dir, language='EN', brain_region=False, ignoreOOV = None):
        self.region = brain_region
        self.oov = ignoreOOV
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
        succ = False
        while not succ:
            try:
                run = int(self.file_list[index][self.run_index])
                target = self.label_dict[run][index%len(self.label_dict[run])]
                if target == self.oov:
                    raise IndexError(f"Ignoring index {index} corresponding to oov")
                succ = True
            except:
                index = np.random.choice((range(0, self.__len__())))
        img = self.normalize_data(np.load(self.file_list[index]))
        return img, target
    
    def __len__(self):
        return len(self.file_list)

    def normalize_data(self, data):
        # Data auguentaion
        data = scipy.stats.zscore(data, axis=None)
        data[~ np.isfinite(data)] = 0
        return data
        
    

    
        
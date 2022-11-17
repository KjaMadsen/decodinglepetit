# %%
#Train, val, Test splitter
import os
from sklearn.model_selection import train_test_split
import shutil
import random

# %%
#Structure
#
#data - raw_data 
# |
# run (0,1,2,3,...)
# |
# CN - EN - FR + labels.json
# |
# MRI-files 

unsorted_data_dir = ""

def get_all_data(unsorted_data_dir):
    data = []
    for dirpath,_,files in os.walk(unsorted_data_dir):
                for run, scan in enumerate(sorted(files)):
                    if scan.endswith(".nii.gz"):
                        data.append((run, os.path.join(dirpath, scan)))
    return data

def sort_by_subject(data, add_run=False):
    output = {sub: [] for sub in os.listdir(unsorted_data_dir)}
    for run,file in data:
        if add_run:
            output[file[len(unsorted_data_dir):len(unsorted_data_dir)+9]].append((run,file))
        else:
            output[file[len(unsorted_data_dir):len(unsorted_data_dir)+9]].append(file)
    return {k:v for k,v in output.items() if len(v)>0}

def sort_by_language(data):
    output = {"EN":[], "CN":[], "FR":[]}
    for run,file in data:
        output[file[len(unsorted_data_dir)+4:len(unsorted_data_dir)+6]].append((run,file))
    return output


#config 1: same language, different subjects
#obs: need to make sure that the same subject is not in train/val/test
def config1(data, language = "CN", split=(0.8,0.1,0.1)):
    relevant_files = sort_by_subject(sort_by_language(data)[language], add_run=True)
    random.shuffle(relevant_files)
    split_ = (int(len(relevant_files)*split[0]), int(len(relevant_files)*split[0])+int(len(relevant_files)*split[1])) 
    train = [f for f in list(relevant_files.keys())[:split_[0]]]
    val = [f for f in list(relevant_files.keys())[split_[0]:split_[1]]]
    test = [f for f in list(relevant_files.keys())[split_[1]:]]
    folders = {"Train":train, "Val":val, "Test":test}
    for k, subs in folders.items():
        for sub in subs:
            for run, file in relevant_files[sub]:
                if not os.path.exists(f"data/{k}/{run}/{language}/"):
                    os.makedirs(f"data/{k}/{run}/{language}/")
                shutil.copy(file, f"data/{k}/{run}/{language}/")
    
#config 2: same language, same subject
#use only one subject to train/val/test
#obs: need to make sure the labels exists in train/val/test

def config2(data, language = "CN"):
    relevant_files = sort_by_subject(sort_by_language(data)[language])
    subject = random.choice(list(relevant_files.values())) #choose a random subject for training
    random.shuffle(subject) 
    train = subject[:len(subject)-1]
    val = subject[-2]
    test = subject[-1]
    folders = {"Train":train, "Val":val, "Test":test}
    for k, v in folders.items():
        for file in v:
            if not os.path.exists(f"data/{k}/{subject}"):
                os.makedirs(f"data/{k}/{subject}")
            shutil.copy(file, f"data/{k}/{subject}/")
    
#config 3: different language
def config3(data, train_language = "CN", test_language = "EN"):
    train_files = sort_by_language(data)[train_language]
    test_files = sort_by_language(data)[test_language]

    for run, file in train_files:
        if not os.path.exists(f"data/Train/{run}/{train_language}/"):
                os.makedirs(f"data/Train/{run}/{train_language}/")
        shutil.copy(file, f"data/Train/{run}/{train_language}/")

    val = test_files.items()[:len(test_files)/2]
    test = test_files.items()[len(test_files)/2:]
    for run, file in val:
        if not os.path.exists(f"data/Val/{run}/{test_language}/"):
                os.makedirs(f"data/Val/{run}/{test_language}/")
        shutil.copy(file, f"data/Val/{run}/{test_language}/")
    for run, file in test:
        if not os.path.exists(f"data/Test/{run}/{test_language}/"):
                os.makedirs(f"data/Test/{run}/{test_language}/")
        shutil.copy(file, f"data/Test/{run}/{test_language}/")
        
def main():
    unsorted_data_dir = "raw_data/derivatives/"
    #split = (0.8,0.1,0.1)
    random.seed(1234)
    #__init__
    for dirpath,s,files in os.walk("data/"):
        for run, scan in enumerate(sorted(files)):
            if scan.endswith(".nii.gz"): #maybe include labels?
                os.remove(os.path.join(dirpath, scan))
    
    data = get_all_data(unsorted_data_dir)
    # config1(data, language="CN")
    # config2(data, language="CN")
    # config3(data, train_language="CN", test_language="EN")
    return

if __name__=="__main__":
    main()

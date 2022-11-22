#Train, val, Test splitter
import os
import shutil
import random
import textgrid as txt
import numpy as np
#Structure
#
# data - raw_data/derivatives
# |
# Train - Val - Test
# |
# run (0,1,2,3,...)
# |
# CN - EN - FR
# |
# MRI-files and labels.txt

def shuffle_dict(dictionary):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    shuffled_dict = {}
    for key in keys:
        shuffled_dict.update({key: dictionary[key]})
    return shuffled_dict

def get_all_data(unsorted_data_dir):
    data = []
    for dirpath,_,files in os.walk(unsorted_data_dir):
                for run, scan in enumerate(sorted(files)):
                    if scan.endswith(".nii.gz"):
                        data.append((run, os.path.join(dirpath, scan)))
    return data

def sort_by_subject(data, unsorted_data_dir, add_run=False):
    output = {sub: [] for sub in os.listdir(unsorted_data_dir)}
    for run,file in data:
        if add_run:
            output[file[len(unsorted_data_dir):len(unsorted_data_dir)+9]].append((run,file))
        else:
            output[file[len(unsorted_data_dir):len(unsorted_data_dir)+9]].append(file)
    return {k:v for k,v in output.items() if len(v)>0}

def sort_by_language(data, unsorted_data_dir):
    output = {"EN":[], "CN":[], "FR":[]}
    for run,file in data:
        output[file[len(unsorted_data_dir)+4:len(unsorted_data_dir)+6]].append((run,file))
    return output


#config 1: same language, different subjects
#obs: need to make sure that the same subject is not in train/val/test
def config1(data, unsorted_data_dir, language = "CN", split=(0.8,0.1,0.1)):
    relevant_files = sort_by_subject(sort_by_language(data, unsorted_data_dir)[language], unsorted_data_dir, add_run=True)
    relevant_files = shuffle_dict(relevant_files)
    split_ = (int(len(relevant_files)*split[0]), int(len(relevant_files)*split[0])+int(len(relevant_files)*split[1])) 
    train = [f for f in list(relevant_files.keys())[:split_[0]]]
    val = [f for f in list(relevant_files.keys())[split_[0]:split_[1]]]
    test = [f for f in list(relevant_files.keys())[split_[1]:]]
    folders = {"Train":train, "Val":val, "Test":test}
    for partition, subs in folders.items():
        for sub in subs:
            for run, file in relevant_files[sub]:
                if not os.path.exists(f"data/{partition}/{run}/{language}/"):
                    os.makedirs(f"data/{partition}/{run}/{language}/")
                shutil.copy(file, f"data/{partition}/{run}/{language}/")
                print(file)
    
#config 2: same language, same subject
#use only one subject to train/val/test
#obs: need to make sure the labels exists in train/val/test
def config2(data, unsorted_data_dir, language = "CN"):
    relevant_files = sort_by_subject(sort_by_language(data, unsorted_data_dir)[language], unsorted_data_dir)
    subject_name = random.choice(list(relevant_files.keys())) #choose a random subject for training
    subject = relevant_files[subject_name]
    random.shuffle(subject) 
    train = subject[:len(subject)-1]
    val = [subject[-2]]
    test = [subject[-1]]
    folders = {"Train":train, "Val":val, "Test":test}
    for k, v in folders.items():
        for file in v:
            if not os.path.exists(f"data/{k}/{subject_name}"):
                os.makedirs(f"data/{k}/{subject_name}")
            shutil.copy(file, f"data/{k}/{subject_name}/")
            print(file)
    
#config 3: different language
def config3(data, unsorted_data_dir, train_language = "CN", test_language = "EN"):
    train_files = sort_by_language(data, unsorted_data_dir)[train_language]
    test_files = sort_by_language(data, unsorted_data_dir)[test_language]

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

def prepare_labels(textgrid_dir, destination_dir):
    for path in textgrid_dir:
        file = txt.TextGrid.fromFile(path)
        # adds words that happen in transistion between scans
        # to both scans it appears in
        result = []
        tmp = []
        for _, i in enumerate(range(len(file[0]))):
            j = file[0][i].maxTime
            if (j-len(result)*2) <= 2:
                tmp.append(file[0][i].mark)
            else:
                if file[0][i].minTime-len(result)*2 <= 2:
                    o = file[0][i].mark
                    tmp.append(o)
                result.append(tmp)
                tmp = [o]
        with open("labels.txt", "w") as file:
            file.writelines


def main():
    unsorted_data_dir = "raw_data/derivatives/"
    #split = (0.8,0.1,0.1)
    random.seed(1234)
    #__init__
    print("\nRemoving folders...")
    for i in os.listdir("data/"):
        shutil.rmtree("data/"+i)
    
    #for dirpath,s,files in os.walk("data/"):
    #    for run, scan in enumerate(sorted(files)):
    #        if scan.endswith(".nii.gz"): #maybe include labels?
    #            os.remove(os.path.join(dirpath, scan)) #clear train/val/test folders
    
    print("Done removing folders")
    print("\nRetriving data...")
    data = get_all_data(unsorted_data_dir)
    print("Done retriving data")
    print("Moving data into folders....")
    # config1(data, unsorted_data_dir, language="CN")
    # config2(data, unsorted_data_dir, language="CN")
    # config3(data, train_language="CN", test_language="EN")
    print("Done!")
    return 0

if __name__=="__main__":
    main()

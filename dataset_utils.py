#Train, val, Test splitter
import os
import shutil
import random
import numpy as np
import pandas as pd
from copy import copy
import nibabel as nib
import codecs
import json
from sklearn.model_selection import train_test_split
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

RANDOM_SEED = 1234

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
    language_dict = {"EN":[], "CN":[], "FR":[]}
    for run,file in data:
        language_dict[file[len(unsorted_data_dir)+4:len(unsorted_data_dir)+6]].append((run,file))
    return language_dict

def words_to_labels(result):
    labels_dict = {k:v for k,v in enumerate(set([word for sublist in result.values() for word in sublist]))}
    
    #this chunk ensures that oov has the highest label so it can more easily be ignored
    tmp_w = labels_dict[len(labels_dict.keys())-1]
    labels_dict = {v:k for k,v in labels_dict.items()}
    tmp_k = labels_dict["oov"]
    labels_dict["oov"] = len(labels_dict.keys())-1
    labels_dict[tmp_w] = tmp_k
        
    labels = {k:[] for k in range(9)}
    for run, words in result.items():
        for word in words:
            labels[run].append(labels_dict[word])
    save_labels_dict(labels_dict, "labels.json")
    print("number of classes = ", len(list(labels_dict.values())))
    return labels

def save_labels_dict(labels_dict, dest_file_path):
    with open(dest_file_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f)

def save_nii_as_npy(file, destination):
    img = nib.load(file).get_fdata(dtype=np.float32).transpose(3,0,1,2)
    for idx, t in enumerate(img[4:]): #discard first four warmup runs
        np.save(destination + file[35:45] + file[56:63] + f"_{idx}", t)

def inter_subject(data_dir, language = "EN"):
    all_files = []
    for run in os.listdir(data_dir):
        for sub in os.listdir(f"{data_dir}/{run}/{language}"):
            for file in os.listdir(f"{data_dir}/{run}/{language}/{sub}"):
                all_files.append((run, f"{data_dir}/{run}/{language}/{sub}/{file}"))
        
    return all_files
      
def cross_subject(data_dir, language = "EN"):
    all_subjects = []
    for run in os.listdir(data_dir):
        for sub in os.listdir(f"{data_dir}/{run}/{language}"):
            all_subjects.append((run, f"{data_dir}/{run}/{language}/{sub}"))

    return all_subjects
    
def raw2npy(unsorted_data_dir, language = "EN"):
    print("loading data.....")
    relevant_files = sort_by_subject(sort_by_language(get_all_data(unsorted_data_dir), unsorted_data_dir)[language], unsorted_data_dir, add_run=True)
    for sub, data in relevant_files.items():
        for run, file in data:
            if not os.path.exists(f"data/{run}/{language}/{sub}"):
                os.makedirs(f"data/{run}/{language}/{sub}")
            print(file)
            save_nii_as_npy(file, f"data/{run}/{language}/{sub}")
    print("\ndone loading data!\n")

  
def prepare_handpicked_labels(annotation_file, vocab):
    df = pd.read_csv(annotation_file)
    
    # labels = {word:lbl for lbl,word in enumerate(set(vocab))}
    
    result = {k:[] for k in range(9)}
    count = [0]*9
 
    for section in range(9):
        for _, row in df.loc[df["section"] == section+1].iterrows():
            w = row["lemma"]
            off_index = int(row["offset"]/2)
            on_index = int(row["onset"]/2)
            if w in vocab:
                result[section].append((on_index, off_index, w))
        count[section] = off_index
    
    curated_labels = words_to_labels(format_label_dict(result, count))
    return curated_labels
    
def create_label_files(labels_dict, language):
    for partition in ["Train", "Val", "Test"]:
        for run, labels in labels_dict.items():
            file_path = os.path.join(partition, f"data/{run}/{language}/")
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(os.path.join(file_path, "labels.txt"), "w") as f:
                for label in labels:
                    f.write(str(label) + "\n")

def format_label_dict(dictionary, count):
    new_dictionary =  {k:["oov"]*(count[k]+1) for k in range(9)}
    for k, v in dictionary.items():
        for on_idx, off_idx, word in v:
            new_dictionary[k][on_idx] = word
            new_dictionary[k][off_idx] = word
    return new_dictionary
            

def prepare_labels(annotation_file, pos="PRON"):
    df = pd.read_csv(annotation_file)
    result = {k:[] for k in range(9)}
    count = [0]*9
    for section in range(9):
        for _, row in df.loc[df["section"] == section+1].iterrows():
            word = row["lemma"]
            off_index = int(row["offset"]/2)
            on_index = int(row["onset"]/2)
            if row["pos"] == pos:
                result[section].append((on_index, off_index, word))
        count[section] = off_index
    result = format_label_dict(result, count)
    data = words_to_labels(result)

    return data
                    

def clear_data_dir():
    print("\nDelete the contents of data dir?...\n[Y/N]?")
    ui = input()
    if ui.lower() == "y":
        for i in os.listdir("data/"):
            if os.path.isdir("data/" + i):
                shutil.rmtree("data/"+i)
        print("Done removing folders")

def partition_dataset(paths, split):
                
    train, test_val, _, _ = train_test_split(paths, [""]*len(paths), test_size = 1-split[0], random_state = RANDOM_SEED)
    val, test, _, _ = train_test_split(test_val, [""]*len(test_val), test_size = split[2]/(split[1]+split[2]), random_state = RANDOM_SEED)
    
    partitions = [("Train", train), ("Val", val), ("Test", test)]
    for partition, l in partitions:
        for item in l:
            (run, path) = item
            
            if os.path.isdir(path):
                new_path = os.path.join(partition, path)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                for file in os.listdir(path):
                    
                    shutil.copy(os.path.join(path, file), os.path.join(new_path, file))
            else:
                new_path = os.path.join(partition, '/'.join(path.split("/")[:-1]))
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copy(path, os.path.join(partition, path))


def prepare_dataset(config_version, data_dir, annotation_path, split, language, pos):
    if "inter" in config_version:
        files = inter_subject(data_dir, language)
    elif "cross" in config_version:
        files = cross_subject(data_dir, language)
    else:
        raise NotImplementedError(f"The configuration {config_version} is not specified")

    partition_dataset(files, split)
    labels = prepare_labels(annotation_path,
                            pos=pos)

    create_label_files(labels, language)
    


#Train, val, Test splitter
import os
import shutil
import random
import numpy as np
import pandas as pd
from copy import copy
import nibabel as nib
import codecs
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

def words_to_labels(result):
    labels = {v:k for k,v in enumerate(set([word for sublist in result.values() for word in sublist]))}
    output = {k:[] for k in range(9)}
    for run, words in result.items():
        for word in words:
            output[run].append(labels[word])
    make_labels_dict_file(labels)
    print(len(list(labels.values())), " = number of classes")
    return output

def make_labels_dict_file(labels_dict, dir="./"):
    with codecs.open(dir+"label_dict.txt", "w", encoding="utf-8") as f:
        for k,v in labels_dict.items():
            f.writelines(str(k) + " = "+ str(v) + "\n")

def save_nii_as_npy(file, destination):
    img = nib.load(file).get_fdata(dtype=np.float32).transpose(3,0,1,2)
    for idx, t in enumerate(img[4:]): #discard first four
        np.save(destination + file[35:45] + file[56:63] + f"_{idx}", t)
    
        
def config2(data_dir, split=(0.8,0.1,0.1), language = "EN"):
    all_files = []
    for run in range(9):
        for sub in os.listdir(f"{data_dir}/{run}/{language}"):
            for file in os.listdir(f"{data_dir}/{run}/{language}/{sub}"):
                all_files.append((run, f"{data_dir}/{run}/{language}/{sub}/{file}"))
        
        

    train, test_val, _, _ = train_test_split(all_files, [""]*len(all_files), test_size = 1-split[0], random_state = 1234)
    val, test, _, _ = train_test_split(test_val, [""]*len(test_val), test_size = 0.5, random_state = 1234)
    for partition, l in [("Train", train), ("Val", val), ("Test", test)]:
        for run, file in l:
            if not os.path.exists(f"data/{partition}/{run}/{language}"):
                os.makedirs(f"data/{partition}/{run}/{language}")
            shutil.move(file, f"data/{partition}/{run}/{language}")
            
    for run in range(9):
        shutil.rmtree(f"data/{run}")
        #os.remove(f"data/{run}")
               

def load_data(unsorted_data_dir, language = "EN"):
    relevant_files = sort_by_subject(sort_by_language(get_all_data(unsorted_data_dir), unsorted_data_dir)[language], unsorted_data_dir, add_run=True)
    for sub, data in relevant_files.items():
        for run, file in data:
            if not os.path.exists(f"data/{run}/{language}/{sub}"):
                os.makedirs(f"data/{run}/{language}/{sub}")
            print(file)
            save_nii_as_npy(file, f"data/{run}/{language}/{sub}")
    

def prepare_spare_classes(annotation_file, destination_dir, target_words, language="EN", oov=""):
    # target_words = {"me" : 0, "you": 1, "myself": 0}
    make_labels_dict_file(target_words)
    file = pd.read_csv(annotation_file)
    result = {k:[] for k in range(9)}
    count = 0
    num_of_classes = len(set(target_words.values()))
    sec = 1
    for _, row in file.iterrows():
        if row["section"] != sec:
            count = 0
        sec = row["section"]
        w = row["lemma"]
        if (row["onset"]-count*2) < 2:
            count +=1  
            if w in target_words.keys():
                result[sec-1].append(target_words[w])
            else:
                result[sec-1].append(oov)

    for n in ["Train", "Val", "Test"]:
        for run,v in result.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for word in v:
                    f.write(str(word) + "\n")
                    
    return num_of_classes


def convert_to_binary_labels(destination_dir, oov, language = "EN"):
    for n in ["Train", "Test", "Val"]:
        for run in range(9):
            f = np.loadtxt(destination_dir+ f"{n}/{run}/{language}/labels.txt")
            with open("label_dict.txt", "r") as lbls:
                for lbl in lbls.readlines():
                    w, l = lbl.split("=")
                    if w.strip() == oov:
                        oov_i = int(l.strip("\n"))
            binary_labels = np.array(f!=oov_i, dtype=int)
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as nf:
                for i in binary_labels:
                    nf.write(str(i)+"\n")
    return True


def prepare_handpicked_labels(annotation_file, destination_dir, vocab, language = "EN", oov=""):
    df = pd.read_csv(annotation_file)
    labels = {word:lbl for lbl,word in enumerate(set(vocab))}
    make_labels_dict_file(labels)
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
    
    print(reformat_messy_dict(result, oov, count))
    output = words_to_labels(reformat_messy_dict(result, oov, count))
    
    for n in ["Train", "Val", "Test"]:
        for run,v in output.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for word in v:
                    f.write(str(word) + "\n")

def reformat_messy_dict(dictionary, oov, count):
    new_dictionary =  {k:[oov]*(count[k]+1) for k in range(9)}
    for k, v in dictionary.items():
        for i in v:
            on_idx, off_idx, w = i
            new_dictionary[k][on_idx] = w
            new_dictionary[k][off_idx] = w
    return new_dictionary
            

def prepare_labels(annotation_file, destination_dir, language = "EN", pos="PRON", oov=""):
    df = pd.read_csv(annotation_file)
    # adds words that happen in transistion between scans
    # to both scans it appears in
    result = {k:[] for k in range(9)}
    sec = 1
    count = [0]*9
    for section in range(9):
        for _, row in df.loc[df["section"] == section+1].iterrows():
            w = row["lemma"]
            off_index = int(row["offset"]/2)
            on_index = int(row["onset"]/2)
            if row["pos"] == pos:
                result[section].append((on_index, off_index, w))
        count[section] = off_index
    print(reformat_messy_dict(result, oov, count))
    output = words_to_labels(reformat_messy_dict(result, oov, count))
    
    for n in ["Train", "Val", "Test"]:
        for run,v in output.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for word in v:
                    f.write(str(word) + "\n")
                    

def clear_data_dir():
    print("\nDelete the contents of data dir?...\n[Y/N]?")
    ui = input()
    if ui.lower() == "y":
        for i in os.listdir("data/"):
            if os.path.isdir("data/" + i):
                shutil.rmtree("data/"+i)
        print("Done removing folders")

def fill_data_dir(unsorted_data_dir, config, language):
    print("\nRetriving data...")
    data = get_all_data(unsorted_data_dir)
    print("Done retriving data")
    print("Moving data into folders....")
    config(data, unsorted_data_dir, language=language)
    
    
def main():
    language = "EN"
    unsorted_data_dir = "raw_data/derivatives/"
    annotation_file = f"raw_data/annotation/{language}/lppEN_word_information.csv"
    
    #split = (0.8,0.1,0.1)
    random.seed(1234)
    #__init__
    print("\nCopy raw data to data dir?...\n[Y/N]?")
    ui = input()
    if ui.lower() == "y":
        for i in os.listdir("data/"):
            if os.path.isdir("data/" + i):
                shutil.rmtree("data/"+i)
        print("Done removing folders")

       

        print("\nRetriving data...")
        data = get_all_data(unsorted_data_dir)
        print("Done retriving data")
        
        print("Moving data into folders....")
        config1(data, unsorted_data_dir, language=language)
        # config2(data, unsorted_data_dir, language="EN")
        # config3(data, train_language="CN", test_language="EN")
    oov = "-1" #label to assign to 'out of vocabulary' words
    print("\npreparing labels")
    prepare_labels(annotation_file, "data/", language, pos="NOUN", oov=oov)
    #prepare_binary_labels("data/", oov, language=language)
    #vocab = ["picture", "forest", "bridge", "golf"]
    #prepare_handpicked_labels(annotation_file, "data/", vocab, language="EN", oov=oov)
    # prepare_dummy_labels(annotation_file, "data/", language="EN")
    # target_words = {"me":0, "you":1, "myself":0, "i":0, "yourself":1}
    # prepare_spare_classes(annotation_file, "data/", target_words, language="EN", oov=oov)
    
    print("Done with labels")
    print("Done!")

    return 0

if __name__=="__main__":
    load_data("raw_data/derivatives/")
    config2("data/")

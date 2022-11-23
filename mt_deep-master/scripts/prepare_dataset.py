#Train, val, Test splitter
import os
import shutil
import random
import textgrid as txt
import numpy as np
import pandas as pd
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
    return output

def make_labels_dict_file(labels_dict, dir="./"):
    with open(dir+"label_dict.txt", "w") as f:
        for k,v in labels_dict.items():
            f.writelines(str(k) + " = "+ str(v) + "\n")

#config 1: same language, different subjects
#obs: need to make sure that the same subject is not in train/val/test
def config1(data, unsorted_data_dir, language = "CN", split=(0.8,0.1,0.1)):
    relevant_files = sort_by_subject(sort_by_language(data, unsorted_data_dir)[language], unsorted_data_dir, add_run=True)
    relevant_files = shuffle_dict(relevant_files)
    split_ = (int(len(relevant_files)*split[0]), int(len(relevant_files)*split[0])+int(len(relevant_files)*split[1])) 
    train = list(relevant_files.keys())[:split_[0]]
    val = list(relevant_files.keys())[split_[0]:split_[1]]
    test = list(relevant_files.keys())[split_[1]:]
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

def prepare_spare_classes(annotation_file, destination_dir, target_words, language="EN"):
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
                result[sec-1].append("")

    for n in ["Train", "Val", "Test"]:
        for run,v in result.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for word in v:
                    f.write(str(word) + "\n")
                    
    return num_of_classes


def prepare_dummy_labels(annotation_file, destination_dir, language = "EN"):
    file = pd.read_csv(annotation_file)
    result = {k:[] for k in range(9)}
    for _, row in file.iterrows():
        result[int(row["section"])-1].append(row["idx"])

    for n in ["Train", "Val", "Test"]:
        for run, content in result.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for kk in content:
                    # for word in w:
                    #     f.write(word + " ")
                    f.write(str(kk)+"\n") 


def prepare_handpicked_labels(annotation_file, destination_dir, vocab, language = "EN"):
    file = pd.read_csv(annotation_file)
    labels = {word:lbl for lbl,word in enumerate(set(vocab))}
    make_labels_dict_file(labels)
    result = {k:[] for k in range(9)}
    count = 0
    sec = 1
    for _, row in file.iterrows():
        if row["section"] != sec:
            count = 0
        sec = row["section"]
        w = row["lemma"]
        if (row["onset"]-count*2) < 2:
            count +=1  
            if w in vocab :
                result[sec-1].append(labels[w])
            else:
                result[sec-1].append("")
    
    for n in ["Train", "Val", "Test"]:
        for run,v in result.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for word in v:
                    f.write(str(word) + "\n")

def prepare_labels(annotation_file, destination_dir, language = "EN", pos="PRON"):
    file = pd.read_csv(annotation_file)
    # adds words that happen in transistion between scans
    # to both scans it appears in
    result = {k:[] for k in range(9)}
    count = 0
    sec = 1
    for _, row in file.iterrows():
        if row["section"] != sec:
            count = 0
        sec = row["section"]
        w = row["lemma"]
        if (row["onset"]-count*2) < 2:
            count +=1  
            if row["pos"] == "PRON":
                result[sec-1].append(w)
            else:
                result[sec-1].append("")
    output = words_to_labels(result)
    for n in ["Train", "Val", "Test"]:
        for run,v in output.items():
            if not os.path.exists(destination_dir+ f"{n}/{run}/{language}/"):
                os.makedirs(destination_dir+ f"{n}/{run}/{language}/")
            with open(destination_dir+ f"{n}/{run}/{language}/labels.txt", "w") as f:
                for word in v:
                    f.write(str(word) + "\n")
                     


def main():
    unsorted_data_dir = "raw_data/derivatives/"
    annotation_file = "raw_data/anno/annotation-EN-lppEN_word_information.csv"
    language = "EN"
    #split = (0.8,0.1,0.1)
    random.seed(1234)
    #__init__
    print("\nRemoving folders...")
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

    print("\npreparing labels")
    # prepare_labels(annotation_file, "data/", language, pos="PRON")
    # vocab = ["picture", "forest", "bridge", "golf"]
    # prepare_handpicked_labels(annotation_file, "data/", vocab, language="EN")
    #prepare_dummy_labels(annotation_file, "data/", language="EN")
    # target_words = {"me":0, "you":1, "myself":0, "i":0, "yourself":1}
    # prepare_spare_classes(annotation_file, "data/", target_words, language="EN")
    
    print("Done with labels")
    print("Done!")
    
    return 0

if __name__=="__main__":
    main()

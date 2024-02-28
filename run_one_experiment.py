"""
Run one iteration of the experiment, training on one language and testing on another.
"""
import argparse
import csv
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
from transformers import BertTokenizer, BertModel

from utils import train_classifier, eval_classifier, eval_classifier_ood
import data
#import reporter



# The size to cap the training data. Size is measured in cased nouns.
# We chose the number of cased nouns in Basque as our limit.


train_size = 10
test_size = 10

d_lang = {"fr":"UD_French-PUD"}

model = None
tokenizer = None
num_layers = 12
data_path = "../ud-treebanks-v2.13"   
reeval_src_test = True
classifier_path = [0,1,2,3,4,5,6,7,8,9,10,11,12] #classifier_path[i] is path of the classifier of layer i
lang = "fr"
dest_path = ""
load_bert = True
use_saved_files = False

def run_experiment(d_lang,lang,train_size,test_size,classifier_path,use_saved_files=True,load_bert=True):
    if load_bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                        output_hidden_states=True)
        model.eval()
        num_layers = model.config.num_hidden_layers
    else:
        tokenizer = None
        model = None
        num_layers = 12


    has_trained_classifiers = all([os.path.exists(path) for path in classifier_path])
    b = has_trained_classifiers and use_saved_files

    if b:
        print("Classifiers already trained!")

    if not b:
        train_classifiers(d_lang,lang,classifier_path, model, tokenizer, train_size,test_size,use_saved_files, load_bert)

    exit()
    

    if reeval_src_test:
        print(f"Loading the source test set, with limit {test_size}")
        src_test = data.CaseDataset(
            d_lang[lang] +"/"+d_lang[lang].lower() +"-test.conllu", model, tokenizer,
            load_bert,use_saved_files)
    print(f"Loading the dest test set, with limit {test_size}")
    dest_test = data.CaseDataset( dest_path,model, tokenizer, load_bert,use_saved_files)

    out_df = pd.DataFrame([])
    # Layers trained in reverse so we can make sure code is working with informative layers early
    for layer in reversed(range(num_layers+1)):
        print("On layer", layer)
        classifier_path = classifier_path[layer]
        classifier, labelset, labeldict, src_test_accuracy, training_case_distribution = pickle.load(open(classifier_path, "rb"))
        print(f"Loaded case classifier from {classifier_path}!")
        print("src_test_accuracy:", src_test_accuracy)
        if args.reeval_src_test:
            src_test_dataset = data.CaseLayerDataset(src_test, layer_num=layer, labeldict=labeldict)
            src_test_accuracy = eval_classifier(classifier, src_test_dataset)
            print("src_test_accuracy [re-eval]:", src_test_accuracy, "Saving new src test accuracy")
            with open(classifier_path, 'wb') as pkl_file:
                pickle.dump((classifier, labelset, labeldict, src_test_accuracy), pkl_file)
        dest_test_dataset = data.CaseLayerDataset(dest_test, layer_num=layer, labeldict=labeldict)
        print("There are", len(dest_test_dataset), "examples to evaluate on.")
        results = eval_classifier_ood(classifier, labelset, dest_test_dataset)
        results["layer"] = layer
        for key in src_test_accuracy.keys():
            results[f"source_test_accuracy_{key}"] = src_test_accuracy[key]
        print(results)
        out_df = pd.concat((out_df, results), ignore_index=True)

    out_df.to_csv(os.path.join("results", args.output_fn))

def train_classifiers(d_lang,lang, classifier_paths, model, tokenizer, train_size,test_size,use_saved_files,load_bert):
    print("Need to train classifiers!")
    print(f"Loading the source train set, with limit {train_size}")
    path = d_lang[lang] +"/"+d_lang[lang].lower()
    src_train = data.CaseDataset( path+ "-train.conllu",
        model, tokenizer, load_bert,use_saved_files)
    
    #training_case_distribution = src_train.get_case_distribution() #to see
    print(f"Length of train set is {len(src_train)}, limit is {train_size}")
    #if len(src_train) < train_size:
    #    print("Too small! Exiting")
    #    sys.exit()
    src_test = data.CaseDataset(path + "-test.conllu", model, tokenizer, load_bert,use_saved_files)
    if load_bert:
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = 12

    for layer in reversed(range(num_layers+1)):
        classifier_path = classifier_paths[layer]
        if use_saved_files and os.path.exists(classifier_path):
            continue
        train_dataset = data.CaseLayerDataset(src_train, layer_num=layer) 
        print("train dataset labeldict", train_dataset.labeldict) 
        print("Training on", len(train_dataset), "data points.")
        
        if load_bert:
            classifier = train_classifier(train_dataset) 
            print("Trained a case classifier!")
        exit()

        src_test_dataset = data.CaseLayerDataset(src_test, layer_num=layer, labeldict=train_dataset.labeldict)
        src_test_accuracy = eval_classifier(classifier, src_test_dataset) 
        print(f"Accuracy on test set of training language: {src_test_accuracy}")
        print(f"Saving classifier to {classifier_path}")
        with open(classifier_path, 'wb') as pkl_file:
            pickle.dump((classifier, train_dataset.get_label_set(), train_dataset.labeldict, src_test_accuracy, training_case_distribution), pkl_file)

"""
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-lang-base-path', type=str,
        default="/u/nlp/data/dependency_treebanks/UD/2.5/UD_Greek-GDT/el_gdt-ud",
        help="The path to the UD treebanks of the language we're training the classifier on. The path should be without the '-train.connlu'/'-test.conllu' part")
    parser.add_argument('--test-lang-fn', type=str,
        default="/u/nlp/data/dependency_treebanks/UD/2.5/UD_English-PUD/en_pud-ud-test.conllu",
        help="The path to the UD treebank file we're testing the classifier on")
    parser.add_argument('--only-ao', action="store_true",
                        help="When this option is set, the classifier is trained only on A and O nouns (no S to give away alignment)")
    parser.add_argument('--balance', action='store_true', 
                        help="When this option is set, ")
    parser.add_argument("--nom-acc", action="store_true", help="Only train on Nom,Acc nouns")
    parser.add_argument("--erg-abs", action="store_true", help="Only train on Erg,Abs nouns")
    parser.add_argument("--all-major-cases", action="store_true", help="Only train on Nom,Acc,Erg,Abs nouns")
    parser.add_argument('--average-embs', action='store_true', help='With this option, use the average embedding of the subwords of a word, rather than the first subword')
    parser.add_argument("--output-fn", type=str, default="last_run",
                        help="Where to save this run's output")
    parser.add_argument("--reeval-src-test", action="store_true",
                        help="Reevaluate the test set of the source language")
    parser.add_argument("--seed", type=int, default=-1, help="random seed")

    args = parser.parse_args()

    print("args:", args)

    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Just set the seed to {args.seed}")
    else:
        print("Not setting random seed")

"""

run_experiment(d_lang,lang,train_size,test_size,classifier_path,use_saved_files,load_bert)


#if __name__ == "__main__":
#    __main__()

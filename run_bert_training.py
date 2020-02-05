import pandas as pd
import pickle
from tqdm import tqdm
import os

import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import random
import os
import glob

import argparse


from utils import (InputExample, InputFeatures, _truncate_seq_pair, Args,
                      convert_examples_to_features, load_and_cache_examples, train, evaluate)

from pytorch_transformers import (
    WEIGHTS_NAME, BertConfig,
    BertForSequenceClassification,
    BertTokenizer
)

parser = argparse.ArgumentParser()
parser.add_argument("--n_folds", type=int, default=10, help="number of folds for stratified k fold (default 10)")
parser.add_argument("--stop_at", type=int, default=10, help="number of folds to actually compute (default 10)")
parser.add_argument("--train", action="store_true", help="perform training")
parser.add_argument("--test", action="store_true", help="perform testing (requires training data)")
parser.add_argument("--results", action="store_true", help="compute metrics (requires testing data)")
parser.add_argument("--nocuda", action="store_true", help="run all code on cpu")
parser.add_argument("--out_dir", default="BERT_models", help="set root directory for all results (default ./BERT_models)")
parser.add_argument("--level", default="Chapter", choices=["Chapter", "Block", "Category", "Leaf"], help="choose aggregation level for data (default Chapter)")
parser.add_argument("--overwrite", action="store_true", help="ignore existing data and overwrite everything with no additional warning")
parser.add_argument("--small", action="store_true", help="test on a smaller dataset (first 1000 samples)")


line_args = parser.parse_args()
if line_args.stop_at < line_args.n_folds:
    print("*****************\n\n\tWARNING:\n\n\tthe code is going to run on {} out of {} folds!\n\n*****************".format(
    line_args.stop_at, line_args.n_folds))

small = line_args.small
# All important paths and constants


folds_number = line_args.n_folds
stop_after = line_args.stop_at

args = Args()
args.do_train = line_args.train
args.do_eval = line_args.test
args.do_results = line_args.results
args.use_cuda = not line_args.nocuda
args.small = line_args.small
# aggregation_level = "Chapter"
# aggregation_level = "Block"
# aggregation_level = "Category"
# aggregation_level = "Leaf"
aggregation_level = line_args.level

main_dir = "/mnt/HDD/bportelli/lab_avanzato"

original_data_path = "/mnt/HDD/bportelli/lab_avanzato/beatrice.pkl"

diagnosis_df_preprocessed_serialized = main_dir+"/input_df_dropped{}.pkl".format("_small" if small else "")
models_path = main_dir+"/"+line_args.out_dir+"/"
model_name = args.model_name_or_path + "_small" if small else args.model_name_or_path
model_directory = "{}{}/{}/".format(models_path, model_name, aggregation_level)
model_directory_estimators = "{}{}/{}/Estimators/".format(models_path, model_name, aggregation_level)
model_directory_training = "{}{}/{}/Training/".format(models_path, model_name, aggregation_level)
model_directory_predictions = "{}{}/{}/Predictions/".format(models_path, model_name, aggregation_level)

if not os.path.exists(models_path):
    os.mkdir(models_path)
if not os.path.exists(model_directory):
    os.makedirs(model_directory, exist_ok=True)
if not os.path.exists(model_directory_estimators):
    os.mkdir(model_directory_estimators)
if not os.path.exists(model_directory_training):
    os.mkdir(model_directory_training)
if not os.path.exists(model_directory_predictions):
    os.mkdir(model_directory_predictions)

all_features_path = model_directory+"all_features.pkl"
all_label_codes_path = model_directory+"all_label_codes.pkl"
label_map_path = model_directory+"label_map.pkl"
evaluation_path = model_directory+"evaluation_results.pkl"

# print(models_path)
# print(model_directory)
# print(model_directory_estimators)
# print(model_directory_training)
# print(model_directory_predictions)

with open(original_data_path, "rb") as o:
    input_df = pickle.load(o)

print("Original dataframe")
input_df.head()

# ## Let us see how diagnosis are distributed among each:
# - Leaf
# - Category
# - Block
# - Chapter

def compute_relative_frequency(dataframe, label):
    dataframe_freq = dataframe[label].value_counts().to_frame().reset_index().rename(columns={'index':label, label: 'Frequency-Absolute'})
    dataframe_freq["Frequency-Relative"] = dataframe_freq["Frequency-Absolute"].div(len(dataframe))
    dataframe_freq["Frequency-Relative(%)"] = dataframe_freq["Frequency-Absolute"].div(len(dataframe))*100
    return dataframe_freq

diagnosis_df_annotated = input_df

df_leaf_freq = compute_relative_frequency(diagnosis_df_annotated, "Leaf")
df_category_freq = compute_relative_frequency(diagnosis_df_annotated, "Category")
df_block_freq = compute_relative_frequency(diagnosis_df_annotated, "Block")
df_chapter_freq = compute_relative_frequency(diagnosis_df_annotated, "Chapter")

print("Relative/Absolute frequencies for Chapters")
df_chapter_freq

if os.path.exists(diagnosis_df_preprocessed_serialized) and not line_args.overwrite:
    diagnosis_df_annotated = pd.read_pickle(diagnosis_df_preprocessed_serialized)
    print("Length before: {}".format(len(diagnosis_df_annotated)))
    print("Dataframe preprocessed loaded from path: {}".format(diagnosis_df_preprocessed_serialized))
    print("Length after: {}".format(len(diagnosis_df_annotated)))
else:
    for index, row in tqdm(diagnosis_df_annotated.iterrows(), desc="Processed", total=len(diagnosis_df_annotated)):
            text = row["Text"]
            # PRELIMINARY OPERATIONS
            # Removing \n, \r and ° chars
            text = text.replace('\r', ' ')
            text = text.replace('\n', ' ')
            text = text.replace('°', ' ')
            # TOKENIZATION 
            # tokens = nlp(text)
            # PUNCTUATION REMOVAL
            # tokens = [token for token in tokens if token.pos_ != 'PUNCT']
            # LEMMATIZATION
            # lemmas = [t.lemma_ for t in tokens]
            # SAVING PRE PROCESSED TEXT
            # lemmatized_text = ' '.join(lemmas)
            lemmatized_text = text
            diagnosis_df_annotated.at[index,'Text-Processed'] = lemmatized_text

    df_frequency = compute_relative_frequency(diagnosis_df_annotated, aggregation_level)
    print("Filtering entries for a number of folds equal to: {}".format(folds_number))
    print("Length before: {}".format(len(diagnosis_df_annotated)))
    for index, row in df_frequency.iterrows():
        code = row[aggregation_level]
        frequency = row["Frequency-Absolute"]
        if frequency < folds_number:
            diagnosis_df_annotated.drop(diagnosis_df_annotated[diagnosis_df_annotated[aggregation_level] == code].index,inplace=True)
    
    if small:
        diagnosis_df_annotated = diagnosis_df_annotated.iloc[:100,]
    
    print("Length after: {}".format(len(diagnosis_df_annotated)))

    diagnosis_df_annotated.to_pickle(diagnosis_df_preprocessed_serialized)
    print("Dataframe preprocessed saved at path: {}".format(diagnosis_df_preprocessed_serialized))

diagnosis_df_annotated.head()

label_list = input_df[aggregation_level].unique().tolist()
num_labels = len(label_list)

config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

if os.path.exists(all_features_path) and not line_args.overwrite:
    with open(all_features_path, "rb") as o:
        features = pickle.load(o)
    print("All features loaded from path: {}".format(all_features_path))
else:
    text_list = diagnosis_df_annotated["Text-Processed"].values.tolist()

    all_examples = []
    for idx, text in tqdm(enumerate(text_list), desc="Creating examples", total=len(diagnosis_df_annotated)):
        example = InputExample(guid=idx, text_a=text, label=diagnosis_df_annotated.iloc[idx][aggregation_level])
        all_examples.append(example)

    for i in range(3):
        print(all_examples[i])

    features = convert_examples_to_features(all_examples, label_list, args.max_seq_length, tokenizer,
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                args = args)
    with open(all_features_path, "wb") as o:
        pickle.dump(features, o)
    print("All features saved at path: {}".format(all_features_path))
    
for i in range(3):
    print(features[i])

if os.path.exists(all_label_codes_path) and not line_args.overwrite:
    with open(all_label_codes_path, "rb") as o:
        all_label_codes = pickle.load(o)
    with open(label_map_path, "rb") as o:
        label_map = pickle.load(o)
    print("All labels loaded from path: {}".format(all_label_codes_path))

else:
    all_labels = diagnosis_df_annotated[aggregation_level].to_list()
    label_map = {label : i for i, label in enumerate(label_list)}
    all_label_codes = [label_map[l] for l in all_labels]

    del all_labels
    
    with open(all_label_codes_path, "wb") as o:
        pickle.dump(all_label_codes, o)
    with open(label_map_path, "wb") as o:
        pickle.dump(label_map, o)
    print("All labels saved to path: {}".format(all_label_codes_path))
    
print(all_label_codes[0:5])

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.use_cuda :
        torch.cuda.manual_seed_all(args.seed)

del diagnosis_df_annotated

device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
args.device = device
set_seed(args)



print("Current model: {}".format(model_name))
fold_counter = 1

# print(model_directory)
# print(model_directory_estimators)
# print(model_directory_training)

from sklearn.model_selection import StratifiedKFold

codes = all_label_codes

data = {}

sk_fold = StratifiedKFold(n_splits=folds_number)

if args.do_train:

    print("Training")

    for training_indexes, test_indexes in sk_fold.split(X=np.zeros(len(codes)), y=codes):
    
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        model.to(device)
    
        if fold_counter > stop_after:
            print("Stopping before fold: {}".format(fold_counter))
            break
    
        print("Processing fold: {}".format(fold_counter))
        data_serialized = "{}{}-Fold-{}-Data.pkl".format(model_directory_training, model_name, fold_counter)
        
        if os.path.exists(data_serialized) and not line_args.overwrite:
            print("  FOLD ALREADY TRAINED, moving on")
            fold_counter+=1
            continue
        
        # Merging together rows of the CSR matrix selected to be the training set of the current fold
        training_data = []
        training_labels = []
        test_data = []
        test_labels = []
        for index in training_indexes:
            training_data.append(features[index])
            training_labels.append(all_label_codes[index])
        for index in test_indexes:
            test_data.append(features[index])
            test_labels.append(all_label_codes[index])

        # Convert to Tensors and build dataset
        all_unique_ids = torch.tensor([f.unique_id for f in training_data], dtype=torch.long) 
        all_input_ids = torch.tensor([f.input_ids for f in training_data], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in training_data], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in training_data], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in training_data], dtype=torch.long)

        train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_unique_ids)

        train(args, train_dataset, model, tokenizer)

        # Some serialization and things

        data["Fold-{}".format(fold_counter)] = {}
        data["Fold-{}".format(fold_counter)]["Training-Data"] = training_data
        data["Fold-{}".format(fold_counter)]["Training-Labels"] = training_labels
        data["Fold-{}".format(fold_counter)]["Test-Data"] = test_data
        data["Fold-{}".format(fold_counter)]["Test-Labels"] = test_labels
        model_output_dir = "{}{}-Fold-{}-Model".format(model_directory_estimators, model_name, fold_counter)

        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_output_dir)
        torch.save(args, os.path.join(model_output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(model_output_dir)

        print("Model serialized at path: {}".format(model_output_dir))

        if not os.path.exists(data_serialized):
            with open(data_serialized, "wb") as output_file:
                pickle.dump(data, output_file)
                print("Training data serialized at path: {}".format(data_serialized))

        # The fold is processed        

        fold_counter+=1
        
        del model, model_to_save

    # The training is completed

    print("{} training done.".format(model_name))

if args.do_eval:

    print("Evaluating")

    for fold_counter in range(1, folds_number+1):
        print("Processing fold: {}".format(fold_counter))
        data_serialized = "{}{}-Fold-{}-Data.pkl".format(model_directory_training, model_name, fold_counter)
        prediction_serialized = "{}{}-Fold-{}-Prediction.pkl".format(model_directory_predictions, model_name, fold_counter)

        if os.path.exists(prediction_serialized) and not line_args.overwrite:
            print("  FOLD ALREADY TESTED, moving on")
            continue

        model_output_dir = "{}{}-Fold-{}-Model".format(model_directory_estimators, model_name, fold_counter)
        all_predictions_serialized = "{}{}-Fold-{}-All-Predictions.pkl".format(model_directory_predictions, model_name, fold_counter)

        if not os.path.exists(data_serialized): break

        model = model_class.from_pretrained(model_output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

        with open(data_serialized, "rb") as input_file:
            training_data = pickle.load(input_file)

        test_data = training_data["Fold-{}".format(fold_counter)]["Test-Data"]

        # Actual prediction

        # Convert to Tensors and build dataset
        all_unique_ids = torch.tensor([f.unique_id for f in test_data], dtype=torch.long) 
        all_input_ids = torch.tensor([f.input_ids for f in test_data], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_data], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_data], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_data], dtype=torch.long)

        test_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_unique_ids)

        prediction, all_predictions = evaluate(args, test_dataset, model, tokenizer)

        # Some serialization and things

        with open(prediction_serialized, "wb") as output_file:
            pickle.dump(prediction, output_file)
            print("Prediction serialized at path: {}".format(prediction_serialized))
        with open(all_predictions_serialized, "wb") as output_file:
            pickle.dump(all_predictions, output_file)
            print("All-Predictions serialized at path: {}".format(prediction_serialized))

    # The predictions are completed

    print("{} predictions done.".format(model_name))

def compute_accuracy(real_labels, best_preds, all_preds):
    acc_1 = (real_labels == best_preds).mean()
    acc_3 = np.mean([1 if r in a[:3] else 0 for (r,a) in zip(real_labels, all_preds)])
    acc_5 = np.mean([1 if r in a[:5] else 0 for (r,a) in zip(real_labels, all_preds)])
    return acc_1, acc_3, acc_5


if args.do_results:

    print("Calculating Metrics")

    results_df = pd.DataFrame(columns=["Fold", "Accuracy@1", "Accuracy@3", "Accuracy@5"])

    for fold_counter in range(1, folds_number+1):
        print("Processing fold: {}".format(fold_counter))

        data_serialized = "{}{}-Fold-{}-Data.pkl".format(model_directory_training, model_name, fold_counter)
        prediction_serialized = "{}{}-Fold-{}-Prediction.pkl".format(model_directory_predictions, model_name, fold_counter)
        all_predictions_serialized = "{}{}-Fold-{}-All-Predictions.pkl".format(model_directory_predictions, model_name, fold_counter)

        if not os.path.exists(data_serialized): break

        with open(data_serialized, "rb") as input_file:
            data = pickle.load(input_file)
            
        with open(prediction_serialized, "rb") as input_file:
            best_prediction = pickle.load(input_file)
            
        with open(all_predictions_serialized, "rb") as input_file:
            all_predictions = pickle.load(input_file)
        
        real_labels = data["Fold-{}".format(fold_counter)]["Test-Labels"]
        
        acc_1, acc_3, acc_5 = compute_accuracy(real_labels, best_prediction, all_predictions)
        results_df = results_df.append({"Fold": fold_counter,
                           "Accuracy@1": acc_1,
                           "Accuracy@3": acc_3,
                           "Accuracy@5": acc_5}, ignore_index=True)

    results_df.to_pickle(evaluation_path)
    print("Evaluation results saved to path: {}".format(evaluation_path))
    
results_df = pd.read_pickle(evaluation_path)
print(results_df)

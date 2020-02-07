from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import random
import os
import glob
import gc
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import (
    WEIGHTS_NAME, BertConfig,
    BertForSequenceClassification,
    BertTokenizer
)

class Args():
    def __init__(self):
        self.use_cuda = True
        self.seed = 42
        # self.output_dir = "./output_seed_42"
        
        self.do_train = True
        self.do_eval = True
        self.do_results = True
        # self.overwrite_output_dir = False
        self.overwrite_cache = False
        # self.eval_all_checkpoints = False
        # self.evaluate_on_all = True
        
        self.config_name = False
        self.model_name_or_path = "bert-base-multilingual-uncased"
        self.tokenizer_name = False
        self.do_lower_case = True
        # self.task_name = "chapter"
        # self.data_path = "/mnt/HDD/bportelli/lab_avanzato/beatrice.pkl"
        self.small = False
        self.max_seq_length = 256 # MAX 512 because of BERT constraints
        self.cached_dataset_dir = "./cached_seed_42"
        
        # ideal values:
        # 1) *_batch_size=16; gradient_accumulation_steps=1
        # 2) *_batch_size=8; gradient_accumulation_steps=2
        
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.gradient_accumulation_steps = 1

        # limits the length of the training
        # BERT models usually overfit after 4/5 epochs
        self.max_steps = -1
        self.num_train_epochs = 4.0
        self.device = None
        
        self.weight_decay = 0.0 # (default)
        self.learning_rate = 5e-5 # (default)
        self.adam_epsilon = 1e-8 # (default)
        self.max_grad_norm = 1.0 # (default)
        self.warmup_steps = 0 # (default)
        
        self.save_steps = 50

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.use_cuda :
        torch.cuda.manual_seed_all(args.seed)

def compute_accuracy(real_labels, best_preds, all_preds):
    acc_1 = (real_labels == best_preds).mean()
    acc_3 = np.mean([1 if r in a[:3] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    acc_5 = np.mean([1 if r in a[:5] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    acc_10 = np.mean([1 if r in a[:10] else 0 for (r,a) in zip(real_labels, all_preds)]) if all_preds is not None else -1
    return acc_1, acc_3, acc_5, acc_10


def evaluate(args, eval_dataset, model, tokenizer, prefix="", kind="dev"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    results = {}

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    preds_guids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():


            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3] }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            preds_guids = batch[4].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            preds_guids = np.append(preds_guids, batch[4].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # preds.shape = (n,17)
    best_preds = np.argmax(preds, axis=1)
    all_preds_ranked = np.argsort(-preds, axis=1)
    
    results = (best_preds, all_preds_ranked)
    
    return results
    
# get WORD embeddings from BERT model
def get_embeddings(args, model, dataset, kind):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size)
    model.eval()
    
    all_embeddings = None
    
    for batch in tqdm(dataloader, desc="Extracting features from BERT"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():


            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3] }

            outputs = model(**inputs)
            hidden_layers = outputs[2]
            del outputs
            if kind not in ["mean3", "mean5"]:
                lvl = int(kind)
                embeddings = hidden_layers[-lvl]
                
                
            elif kind == "mean5":
                embeddings = hidden_layers[-5:]
                embeddings_sum = embeddings[0]
                for e in embeddings[1:]:
                    embeddings_sum += e
                embeddings = embeddings_sum / 5.0
                
            elif kind == "mean3":
                embeddings = hidden_layers[-3:]
                embeddings_sum = embeddings[0]
                for e in embeddings[1:]:
                    embeddings_sum += e
                embeddings = embeddings_sum / 3.0
            
            embeddings = embeddings.view(embeddings.shape[0], -1) 
            
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat( (all_embeddings, embeddings), dim=0 )

    all_embeddings_cpu = all_embeddings.cpu()
    del all_embeddings
    # torch.cuda.empty_cache()
    return all_embeddings_cpu
    
# get SENTENCE embeddings from BERT model (from [CLS] token)
def get_embeddings_v2(args, model, dataset, kind):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size)
    model.eval()
    
    all_embeddings = None
    
    for batch in tqdm(dataloader, desc="Extracting features from BERT"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():


            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3] }

            outputs = model(**inputs)
            hidden_layers = outputs[2]
            del outputs
            if kind not in ["mean3", "mean5"]:
                lvl = int(kind)
                embeddings = hidden_layers[-lvl][:,0,:]
                
            elif kind == "mean5":
                embeddings = hidden_layers[-5:]
                embeddings_sum = embeddings[0][:,0,:]
                for e in embeddings[1:]:
                    embeddings_sum += e[:,0,:]
                embeddings = embeddings_sum / 5.0
                
            elif kind == "mean3":
                embeddings = hidden_layers[-3:]
                embeddings_sum = embeddings[0][:,0,:]
                for e in embeddings[1:]:
                    embeddings_sum += e[:,0,:]
                embeddings = embeddings_sum / 3.0
            
            embeddings = embeddings.view(embeddings.shape[0], -1) 
            
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat( (all_embeddings, embeddings), dim=0 )
    
    all_embeddings_cpu = all_embeddings.cpu()
    del all_embeddings
    # torch.cuda.empty_cache()
    return all_embeddings_cpu
    
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps )
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)
    
    args.save_steps = 0 # int( len(train_dataset)/ (args.train_batch_size * args.gradient_accumulation_steps) / 2)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3] }
            
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            del outputs

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            
    del optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
            
    if global_step == 0:
        return 1, 1
    else:
        return global_step, tr_loss / global_step



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        
    def __repr__(self):
        return 'InputExample("{}","{}","{}","{}")'.format(self.guid, self.text_a, self.text_b, self.label)
    def __str__(self):
        return 'InputExample("{}","{}","{}","{}")'.format(self.guid, self.text_a, self.text_b, self.label)
    def _repr_pretty_(self, p, cycle):
        return 'InputExample("{}","{}","{}","{}")'.format(self.guid, self.text_a, self.text_b, self.label)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, input_ids, input_mask, segment_ids, label_id,
                 float_mask=None):
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
    def __repr__(self):
        return 'InputFeatures("{}",..., "{}")'.format(self.unique_id, self.label_id)
    def __str__(self):
        return 'InputFeatures("{}",..., "{}")'.format(self.unique_id, self.label_id)
    def _repr_pretty_(self, p, cycle):
        return 'InputFeatures("{}",..., "{}")'.format(self.unique_id, self.label_id)
        
        
    def __eq__(self, other):
        if not isinstance(other, InputFeatures):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.unique_id == other.unique_id and \
               self.input_ids == other.input_ids and \
               self.input_mask == other.input_mask and \
               self.float_mask == other.float_mask and \
               self.segment_ids == other.segment_ids and \
               self.label_id == other.label_id
    
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 args=None,
                                 evaluate=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    
    for (ex_index, example) in enumerate(tqdm(examples, desc="examples to features")):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            special_tokens_count =  3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            special_tokens_count =  2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]
            
        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(unique_id=example.guid,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def load_and_cache_examples(args, task, tokenizer, evaluate=False, all=False):
    
    if all:
        kind = "all"
    elif evaluate:
        kind = "dev"
    else:
        kind = "train"
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.cached_dataset_dir, 'cached_{}_{}_{}_{}{}'.format(
        kind,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        "_small" if args.small else ""))
    print("Looking for "+cached_features_file)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset (will be saved at {})".format(args.cached_dataset_dir))
        label_list = get_labels(task)
        
        if all:
            examples = get_all_examples(task)
        elif evaluate:
            examples = get_test_examples(task)
        else:
            examples = get_train_examples(task)
        print(examples[0])
        
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            args = args,
            evaluate = evaluate
        )
        
        
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_unique_ids = torch.tensor([f.unique_id for f in features], dtype=torch.long) 
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.float)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_unique_ids)
    
    return dataset

def calculate_all_metrics(folds_number, model_directory_training, model_name, model_directory_predictions, evaluation_path):
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
            
        if os.path.exists(all_predictions_serialized):
            with open(all_predictions_serialized, "rb") as input_file:
                all_predictions = pickle.load(input_file)
        else:
            all_predictions = None
        
        real_labels = data["Test-Labels"]
        
        acc_1, acc_3, acc_5, acc_10 = compute_accuracy(real_labels, best_prediction, all_predictions)
        results_df = results_df.append({"Fold": fold_counter,
                           "Accuracy@1": acc_1,
                           "Accuracy@3": acc_3,
                           "Accuracy@5": acc_5,
                           "Accuracy@10": acc_10}, ignore_index=True)

    results_df.to_pickle(evaluation_path)
    print("Evaluation results saved to path: {}".format(evaluation_path))
    
    results_df = pd.read_pickle(evaluation_path)
    print(results_df)
    print()
    print(results_df.mean(axis=0))
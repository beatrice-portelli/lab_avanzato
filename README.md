# lab_avanzato
Elenco e descrizione dei file presenti.


- bert_training.ipynb
- lab_bert.ipynb

- run_bert_training.py
- run_bert_training_lr.py
- run_bert_training_lr_more_features.py
- utils.py

----

# Descrizione

## Informazioni generali
### utils.py
Per **tutti** i file il modello BERT utilizzato dipende dai parametri settati in `utils.py`, all'interno della classe `Args`.

Il modello attualmente selezionato è `bert-base-multilingual-uncased` (trained on **lower-cased** text in the top 102 languages with the largest Wikipedias). Per usare testo **cased** commentare/decomenntare le righe relative `model_name_or_path` e `do_lower_case`.

Il numero di epoche di training è fissato a **4** (`num_train_epochs`).

La batch size è fissata a **16** (`train_batch_size`). Nel caso ci fossero problemi di memoria si consiglia di seguire le indicazioni presenti nei commenti del file (riddure la batch size a 8 e aumentare `gradient_accumulation_steps` per mantenere una batch size virtuale di 16).

## bert_training.ipynb e lab_bert.ipynb
Notebook con celle (anche scollegate tra loro) per visualizzare il dataset, testare la tokenizzazione multilingue di BERT e metodi di estrazione di embedding.

## run_bert_training.py

## run_bert_training_lr.py

## run_bert_training_lr_more_features.py


----

# Usage

## run_bert_training.py
```
usage: run_bert_training.py [-h] [--n_folds N_FOLDS] [--stop_at STOP_AT]
                            [--train] [--test] [--results] [--nocuda]
                            [--out_dir OUT_DIR]
                            [--level {Chapter,Block,Category,Leaf}]
                            [--overwrite] [--small]

optional arguments:
  -h, --help            show this help message and exit
  --n_folds N_FOLDS     number of folds for stratified k fold (default 5)
  --stop_at STOP_AT     number of folds to actually compute (default 5)
  --train               perform training
  --test                perform testing (requires training data)
  --results             compute metrics (requires testing data)
  --nocuda              run all code on cpu
  --out_dir OUT_DIR     set root directory for all results (default
                        ./BERT_models)
  --level {Chapter,Block,Category,Leaf}
                        choose aggregation level for data (default Chapter)
  --overwrite           ignore existing data and overwrite everything with no
                        additional warning
  --small               test on a smaller dataset (first 1000 samples)

```

## run_bert_training_lr.py
```
usage: run_bert_training_lr.py [-h] [--n_folds N_FOLDS] [--stop_at STOP_AT]
                               [--finetune_bert] [--train] [--test]
                               [--results] [--nocuda] [--overwrite]
                               [--out_dir OUT_DIR]
                               [--level {Chapter,Block,Category,Leaf}]
                               [--embedding_lvl {1,2,3,4,5,6,7,8,9,10,11,12,mean3,mean5}]
                               [--embedding_kind {word,sent}] [--small]

optional arguments:
  -h, --help            show this help message and exit
  --n_folds N_FOLDS     number of folds for stratified k fold (default 5)
  --stop_at STOP_AT     number of folds to actually compute (default 5)
  --finetune_bert       fine-tune bert for 4 epochs before using it to
                        generate features for the other models
  --train               perform training
  --test                perform testing (requires training data)
  --results             compute metrics (requires testing data)
  --nocuda              run all code on cpu
  --overwrite           ignore existing data and overwrite everything with no
                        additional warning
  --out_dir OUT_DIR     set root directory for all results (default
                        ./combined_models)
  --level {Chapter,Block,Category,Leaf}
                        choose aggregation level for data (default Chapter)
  --embedding_lvl {1,2,3,4,5,6,7,8,9,10,11,12,mean3,mean5}
                        which of BERT's hidden layers to use for the embedding
                        representation. 1 is the deepest (final output of the
                        model, more context), 12 is the shallowest (word
                        embeddings without context). mean3 and mean5 are the
                        the mean of layers 1-3 and 1-5.
  --embedding_kind {word,sent}
                        choose kind of embeddings: word-level or sentence-
                        level (sentence-level recommended) (default sent)
  --small               test on a smaller dataset (first 1000 samples)
```

## run_bert_training_lr_more_features.py
```
usage: run_bert_training_lr_more_features.py [-h] [--n_folds N_FOLDS]
                                             [--stop_at STOP_AT]
                                             [--finetune_bert] [--train]
                                             [--test] [--results] [--nocuda]
                                             [--overwrite] [--out_dir OUT_DIR]
                                             [--level {Chapter,Block,Category,Leaf}]
                                             [--embedding_lvl {1,2,3,4,5,6,7,8,9,10,11,12,mean3,mean5}]
                                             [--embedding_kind {word,sent}]
                                             [--small]

optional arguments:
  -h, --help            show this help message and exit
  --n_folds N_FOLDS     number of folds for stratified k fold (default 5)
  --stop_at STOP_AT     number of folds to actually compute (default 5)
  --finetune_bert       fine-tune bert for 4 epochs before using it to
                        generate features for the other models
  --train               perform training
  --test                perform testing (requires training data)
  --results             compute metrics (requires testing data)
  --nocuda              run all code on cpu
  --overwrite           ignore existing data and overwrite everything with no
                        additional warning
  --out_dir OUT_DIR     set root directory for all results (default
                        ./combined_models)
  --level {Chapter,Block,Category,Leaf}
                        choose aggregation level for data (default Chapter)
  --embedding_lvl {1,2,3,4,5,6,7,8,9,10,11,12,mean3,mean5}
                        which of BERT's hidden layers to use for the embedding
                        representation. 1 is the deepest (final output of the
                        model, more context), 12 is the shallowest (word
                        embeddings without context). mean3 and mean5 are the
                        the mean of layers 1-3 and 1-5.
  --embedding_kind {word,sent}
                        choose kind of embeddings: word-level or sentence-
                        level (sentence-level recommended) (default sent)
  --small               test on a smaller dataset (first 1000 samples)
  ```

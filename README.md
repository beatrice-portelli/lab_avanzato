Elenco e descrizione dei file presenti.

- bert_training.ipynb
- lab_bert.ipynb
- run_bert_training.py
- run_bert_training_lr.py
- run_bert_training_lr_more_features.py
- utils.py

# Indice
- Informazioni generali
  - utils.py
- bert_training.ipynb e lab_bert.ipynb
- run_bert_training.py
- run_bert_training_lr.py
- run_bert_training_lr_more_features.py
- Usage
  - Parametri generali [IMPORTANTE]
  - run_bert_training.py
  - run_bert_training_lr.py
  - run_bert_training_lr_more_features.py
----

# Informazioni generali
## utils.py
Per **tutti** i file il modello BERT utilizzato dipende dai parametri settati in `utils.py`, all'interno della classe `Args`.

Il modello attualmente selezionato è `bert-base-multilingual-uncased` (trained on **lower-cased** text in the top 102 languages with the largest Wikipedias). Per usare testo **cased** commentare/decomenntare le righe relative `model_name_or_path` e `do_lower_case`.

Il numero di epoche di training è fissato a **4** (`num_train_epochs`). Essendo già pre-trained BERT tende ad andare in overfit dopo 4/5 epoche (si sconsiglia di aumentare).

La batch size è fissata a **16** (`train_batch_size`). Nel caso ci fossero problemi di memoria si consiglia di seguire le indicazioni presenti nei commenti del file (riddure la batch size a 8 e aumentare `gradient_accumulation_steps` per mantenere una batch size virtuale di 16).

# bert_training.ipynb e lab_bert.ipynb
Notebook con celle (anche scollegate tra loro) per visualizzare il dataset, testare la tokenizzazione multilingue di BERT e metodi di estrazione di embedding.

# run_bert_training.py
Usa il modello BERT selezionato per predire una Label in base al testo contenuto nel dataset.

- Input del modello: contenuto colonna "Text" del dataset.
- Label: contenuto della colonna "Leaf"/"Category"/"Block"/"Chapter".
- Output: dato un sample restituisce per ogni possibile valore della Label la probabilità che il sample abbia tale Label. Vengono quindi calcolate le metriche acc, acc@3, acc@5, acc@10.

Gli output sono salvati in `{model_name}/{aggregation_level}/Predictions` nei file:

- `{model_name}-Fold-{fold_counter}-Prediction.pkl` (etichetta della label più probabile)
- `{model_name}-Fold-{fold_counter}-All-Predictions.pkl` (lista di tutte le etichette delle label, in ordine di probabilità crescente)

La corrispondenza label-etichetta è salvata nel dizionario `label_map` in `{model_name}/{aggregation_level}/label_map.pkl`

# run_bert_training_lr.py
Usa il modello BERT selezionato per generare degli embedding da usare come feature per un secondo classificatore (in questo caso Logistic Regression).

Il fine tuning di BERT sul dataset è opzionale (usare `--finetune_bert` per attivarlo). In caso sia scelto viene effettuato come in run_bert_training.py:
- Input di BERT: contenuto colonna "Text" del dataset.
- Label: contenuto della colonna "Leaf"/"Category"/"Block"/"Chapter".
- Output: dato un sample restituisce per ogni possibile valore della Label la probabilità che il sample abbia tale Label.

Gli embedding possono essere estratti a livello di word o sentence (vedi `--embedding_kind`). Ogni word/sentence-embedding ha dimensione 768 (dimensione degli hidden layer del modello), quindi scegliendo i word embedding si ottengono 768*n_words feature. Si è scelto di ricavare i sentence-embedding dalla rappresentazione del primo token dato in input a BERT (token speciale [CLS], usato in fase di training come indicatore per la classificazione della Label).

_(idea alternativa non implementata: sentence-embedding come media dei word-embedding della frase)_

Gli embedding sono basati sugli hidden layer di BERT e si può quindi cambiare la profondità del livello di embedding (vedi `--embedding_lvl`). Il livello 1 è il più profondo (output finale del modello) e quindi contiene più informazioni contestuali. Il livello 12 è il meno profondo (nessuna informazioni contestuale). Il default è una media dei 3 livelli più profondi.

Il training del secondo modello (Logistic Regression) viene effettuato con:
- Input: embedding estratti da BERT.
- Label: contenuto della colonna "Leaf"/"Category"/"Block"/"Chapter".
- Output: Label più probabile. Viene calcolata acc (le altre metriche acc@n vengono restituite con valore -1).

# run_bert_training_lr_more_features.py
Simile a run_bert_training_lr.py.

Usa il modello BERT selezionato per generare degli embedding da usare come feature per un secondo classificatore (in questo caso Logistic Regression). Il secondo classificatore riceve anche feature aggiuntive (sesso, età, ...).

Il fine tuning di BERT sul dataset è opzionale (vedi run_bert_training_lr.py).

Gli embedding vengono ricavati come in run_bert_training_lr.py.

Il training del secondo modello (Logistic Regression) viene effettuato con:
- Input: concatenazione di:
  - valori della colonna "Età" del dataset (normalizzati con MinMaxScaler in intervallo 0-1)
  - valori della colonna "Sesso" del dataset ("M"/"F", codificati poi con OneHotEncoder in modalità automatica)
  - embedding estratti da BERT
- Label: contenuto della colonna "Leaf"/"Category"/"Block"/"Chapter".
- Output: Label più probabile. Viene calcolata acc (le altre metriche acc@n vengono restituite con valore -1).

----

# Usage

## Parametri generali [IMPORTANTE]
Tutti gli script accettano i seguenti parametri (opzionali):
- `--overwrite`: normalmente gli script eseguono il training dei modelli o calcolano risultati solo se necessario (non ricalcolano feature se sono già presenti salvataggi e non ripetono il training d BERT se non è necessario). In questo modo il training può essere interrotto dopo una fold e poi ripreso senza dover ricalcolare tutto. `--overwrite` forza il ricalcolo di tutti i file senza chiedere ulteriori conferme. **Usare solo se si è sicuri di volerlo fare**.
- `--n_folds` e `--stop_at`: numero di fold per StratifiedKFold e dopo quante fold interrompere le operazioni (es: dividi i dati secondo 3 fold, ma esegui computazioni solo sulla prima)
- `--level`: livello di aggregazione (Chapter/Block/...)
- `--train`, `--test`, `--results`: per eseguire solo la fase di training, testing o computazione delle metriche. Ogni passaggio necessita dei file creati dal precedente. Vengono eseguite **solo** le fasi specificate, quindi è necessario specificarne almeno una.
- `--small`: per questioni di test, tronca il dataset ai primi 1000 elementi. L'albero delle cartelle creato è completamente separato da quello normale, quindi non interagisce in alcun modo con i file creati senza `--small`

Gli script che comportano estrazione di embedding hanno i seguenti parametri (opzionali):
- `embedding_kind`: scelta se usare word-embeddings o sentence-embeddings (default sentence, più leggero).
- `embedding_lvl`: scelta del livello di profondità al quale recuperare gli embedding di BERT. Bert ha 12 livelli hidden, quindi i parametri possibili vanno dall'1 (il più profondo) al 12 (il primo livello), più le due opzioni che permettono di fare una media dei 3/5 livelli più profondi. Il default è la media dei 3 livelli più profondi (1,2,3).

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
                        (default mean3) which of BERT's hidden layers to use for the embedding
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
                        (default mean3)  which of BERT's hidden layers to use for the embedding
                        representation. 1 is the deepest (final output of the
                        model, more context), 12 is the shallowest (word
                        embeddings without context). mean3 and mean5 are the
                        the mean of layers 1-3 and 1-5.
  --embedding_kind {word,sent}
                        choose kind of embeddings: word-level or sentence-
                        level (sentence-level recommended) (default sent)
  --small               test on a smaller dataset (first 1000 samples)
  ```

# Implementation of TopicExpan

This is the anonymized github repository for the submission, entitled **"TopicExpan: Topic Taxonomy Expansion via Hierarchy-aware Topic Phrase Generation"**. For reproducibility, the codes and datasets are publicly available during the review phase.

## STEP 1. Install the python libraries / packages

- numpy
- pytorch
- transformers
- dgl
- scikit-learn
- gensim

## Step 2. Download the dataset and GloVe embedding features

- Download the dataset file and place it into your own data directory
  - [Amazon](http://)
  - [DBPedia](http://)
- Download the pretrained embedding file and place it into your own embedding directory
  - [GloVe](https://nlp.stanford.edu/projects/glove/) (`glove.6B.300d.txt` is used in our experiments)

## Step 3. Complete the config files
- Complete the configuration file for each dataset
  - `config_files/config_amazon.json` and `config_files/config_dbpedia.json`
- Update the **target directory paths** in each config file
  - `glove_dir`, `directory` (in both `data_loader_for_training` and `data_loader_for_expansion`), `save_dir`
- Specify several hyperparameters for **the model architecture** and **its optimization** in each config file

## Step 4. Run the code

- **(Preprocessing)** Make `all.pickle` file by preprocessing the initial topic taxonomy and the document corpus
```
python generate_dataset_binary.py --data_dir <data_dir_path>
```
- **(Training step)** Train the neural model for topic-conditional phrase generation
```
python train.py --config <config_file>
```
- **(Expansion step)** Expand the topic taxonomy by generating topic phrases for each virtual topic node 
```
python expand.py --config <config_file> --resume <trained_model_path>
```

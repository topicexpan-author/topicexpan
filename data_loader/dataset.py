import os, time
import re
import pickle
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from gensim.models import KeyedVectors
from transformers import AutoTokenizer


class DatasetBase(Dataset):
    """ Document Dataset BaseClass
    """
    def __init__(self, directory, raw=True, **args):
        """ Raw dataset class for documents
        
        Parameters
        ----------
        directory : str
            data directory
        raw : bool, optional
            load raw dataset from a set of txt files (if True) or load pickled dataset (if False), by default True
        """
        self.corpus = {}  # doc_id -> document raw text
        self.doc2phrases = {} # doc_id -> a list of phrases
        self.topics = {}  # topic_id -> topic surface name
        self.topic_fullhier = {}
        self.topic_feats = []
        self.topic_triples = [] 

        if raw:
            self._load_dataset_raw(directory, **args)
        else:
            self._load_dataset_pickled(directory)


    def _load_dataset_raw(self, directory, **args):
        # inputs
        corpus_suffix = args.get("corpus_suffix", "")
        doc2phrase_suffix = args.get("doc2phrase", "")
        topic_suffix = args.get("topic_suffix", "")
        topic_hier_suffix = args.get("topic_hier_suffix", "")
        topic_feat_suffix = args.get("topic_feat_suffix", "")
        topic_triple_suffix = args.get("topic_triple_suffix", "")
        
        corpus_path = os.path.join(directory, f"corpus{corpus_suffix}.txt")
        doc2phrases_path = os.path.join(directory, f"doc2phrases{doc2phrase_suffix}.txt")
        topic_path = os.path.join(directory, f"topics{topic_suffix}.txt")
        topic_hier_path = os.path.join(directory, f"topic_hier{topic_hier_suffix}.txt")
        topic_feat_path = os.path.join(directory, f"topic_feats{topic_feat_suffix}.txt")
        topic_triples_path = os.path.join(directory, f"topic_triples{topic_triple_suffix}.txt")
        
        # outputs
        output_suffix = args.get("output_suffix", "")
        output_pickle_path = os.path.join(directory, f"all{output_suffix}.pickle")
        
        with open(corpus_path, "r") as fin:
            for line in tqdm(fin, desc="Loading documents"):
                line = line.strip()
                if line:
                    doc_id, doc_text = line.split("\t")
                    self.corpus[doc_id] = doc_text
        
        with open(topic_path, "r") as fin:
            for line in tqdm(fin, desc="Loading topics"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    topic_id = segs[0]
                    topic_raw_name = segs[1]
                    topic_cleaned_name = re.sub(r"[-_]", " ", re.sub(r",", "", topic_raw_name))
                    # topic_cleaned_name = topic_cleaned_name.split('/')[-1]
                    self.topics[topic_id] = topic_cleaned_name

        with open(doc2phrases_path, "r") as fin:
            for line in tqdm(fin, desc="Loading phrases"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    self.doc2phrases[segs[0]] = segs[1:]

        with open(topic_hier_path, "r") as fin:
            for line in tqdm(fin, desc="Loading topic hierarchy"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    parent, childs = segs[0], segs[1].split(",")
                    if parent not in self.topic_fullhier:
                        self.topic_fullhier[parent]  = [child for child in childs]
                    else:
                        self.topic_fullhier[parent] += [child for child in childs]

        print("Loading topic node base features (topic name embeddings) ...")
        self.topic_feats = KeyedVectors.load_word2vec_format(topic_feat_path, binary=False)
        print(f"Finish loading topic embeddings of size {self.topic_feats.vectors.shape}")
        assert self.topic_feats.vectors.shape[0] == len(self.topics) + 1, f"mismatch between number of topics ({len(self.topics)}) and number of topic embeddings ({self.topic_feats.vectors.shape[0]})"

        with open(topic_triples_path, "r") as fin:
            for line in tqdm(fin, desc="Loading positive triples"):
                line = line.strip()
                if line:
                    doc_id, topic_id, phrase_idx = line.split("\t")
                    self.topic_triples.append((int(doc_id), int(topic_id), int(phrase_idx)))

        # save to pickle for faster loading next time
        print("Start saving pickle data")
        with open(output_pickle_path, 'wb') as fout:
            data = {
                "corpus": self.corpus,
                "doc2phrases": self.doc2phrases,
                "topics": self.topics,
                "topic_fullhier": self.topic_fullhier,
                "topic_feats": self.topic_feats,
                "topic_triples": self.topic_triples,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_path}")


    def _load_dataset_pickled(self, directory):
        pickle_path = os.path.join(directory, "all.pickle")
        assert os.path.exists(pickle_path), f"{pickle_path} doesn't exist, run DatasetBase(raw=True) first"
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        
        self.corpus = data["corpus"]
        self.doc2phrases = data["doc2phrases"]
        self.topics = data["topics"]
        self.topic_fullhier = data["topic_fullhier"]
        self.topic_feats = data["topic_feats"]
        self.topic_triples = data["topic_triples"]


class DocTopicPhraseDataset(DatasetBase):
    def __init__(self, directory, batch_type="doc_topic_phrase", len_cutoff=500, alpha=0.2, raw=False):
        super().__init__(directory=directory, raw=raw)
        assert batch_type in ["doc_topic_phrase", "doc_only"]
        self.batch_type = batch_type
        self.len_cutoff = len_cutoff

        self.doc_ids = []
        self.num_topics = len(self.topics.keys())

        self.topicID2topicRank = {topicID: rank for rank, topicID in enumerate(self.topics.keys())} 
        self.topicRank2topicID = {v: k for k, v in self.topicID2topicRank.items()}
        self.num_topics = len(self.topicID2topicRank)

        self.topic_node_feats = []
        for topicRank in sorted(self.topicRank2topicID.keys()):
            topicID = self.topicRank2topicID[topicRank]
            self.topic_node_feats.append(self.topic_feats[topicID])
        self.topic_node_feats = np.array(self.topic_node_feats)
        self.topic_mask_feats = np.array(self.topic_feats['unknown'])        # As masked node feats, it uses 'unknown' word vector 
        # self.topic_mask_feats = np.zeros(self.topic_node_feats.shape[1])   # As masked node feats, it uses 0 vector 

        model_name = "bert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_tokenizer._bos_token = '[unused99]'
        self.bert_tokenizer._eos_token = '[unused100]'
        self.bert_tokenizer.add_tokens(['[unused99]', '[unused100]'], special_tokens=True)
        
        self.topic_invhier = {topicID: [] for topicID in self.topicID2topicRank}
        for topicID, childIDs in self.topic_fullhier.items():
            for childID in childIDs:
                self.topic_invhier[childID].append(topicID)
        
        # For experiments, a portion of leaf topic nodes are randomly deleted
        leaf_topics = [topicRank \
                     for topicID, topicRank in self.topicID2topicRank.items() \
                     if topicID not in self.topic_fullhier and len(self.topic_invhier[topicID]) == 1]
        self.novel_topics = np.random.choice(leaf_topics, int(alpha*len(leaf_topics)), replace=False)
        
        # known parent topic node -> known child topic nodes
        self.topic_hier = {}
        for k, v in self.topic_fullhier.items():
            self.topic_hier[self.topicID2topicRank[k]] = [
                self.topicID2topicRank[topicID] for topicID in v
                if self.topicID2topicRank[topicID] not in self.novel_topics
            ]

        # known parient topic node -> novel child topic nodes
        self.novel_topic_hier = {}
        for k, v in self.topic_fullhier.items():
            self.novel_topic_hier[self.topicID2topicRank[k]] = [
                self.topicID2topicRank[topicID] for topicID in v
                if self.topicID2topicRank[topicID] in self.novel_topics
            ]
        
        self.topic_triples = [
            (doc_id, topic_id, phrase_idx) 
            for doc_id, topic_id, phrase_idx in self.topic_triples
            if topic_id not in self.novel_topics
        ]

        raw_text = []
        topic_freq_weight = np.array([0] * self.num_topics)
        for k, v in tqdm(self.corpus.items()):
            self.doc_ids.append(k)
            raw_text.append(self.corpus[k])
        
        raw_topics = [topic_name for topic_name in self.topics.values()]

        print(f"Tokenizing documents and phrases using {model_name} tokenizer, please wait...")

        self.tokenized_doc = self.bert_tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        self.tokenized_phs = {}
        for k, v in self.doc2phrases.items():
            tokenized_phs = self.bert_tokenizer(v, return_tensors="pt", padding=True, truncation=True, max_length=8)
            tokenized_phs['input_ids'][tokenized_phs['input_ids'] == 101] = self.bert_tokenizer.bos_token_id
            tokenized_phs['input_ids'][tokenized_phs['input_ids'] == 102] = self.bert_tokenizer.eos_token_id
            self.tokenized_phs[k] = tokenized_phs

    def __len__(self):
        if self.batch_type == "doc_only":
            return len(self.doc_ids)
        else:
            return len(self.topic_triples)
    
    def __getitem__(self, idx):
        if self.batch_type == "doc_only":
            doc_id = int(self.doc_ids[idx])
            doc_info = {k: v[idx, :] for k, v in self.tokenized_doc.items()}
            return (doc_id, doc_info)
        else:
            doc_id, topic_id, phrase_idx = self.topic_triples[idx]
            doc_info = {k: v[doc_id, :] for k, v in self.tokenized_doc.items()}
            phrase_info = {k: v[phrase_idx, :] for k, v in self.tokenized_phs[str(doc_id)].items()}
            return (doc_id, doc_info, topic_id, phrase_info)

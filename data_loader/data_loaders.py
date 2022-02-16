from base import BaseDataLoader
from .dataset import DocTopicPhraseDataset
import torch
import torch.nn.functional as F

"""
    DataLoader (and CollateFn) for Training Step
"""

class DocTopicPhraseDataLoader(BaseDataLoader):
    def __init__(self, directory, batch_size=16, shuffle=True, validation_split=0.0, num_workers=1, alpha=0.5):
        self.dataset = DocTopicPhraseDataset(directory=directory, batch_type="doc_topic_phrase", alpha=alpha, raw=False)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=doc_topic_phrase_dataset_collate_fn)
        self.n_samples = len(self.dataset)

def doc_topic_phrase_dataset_collate_fn(samples):
    doc_ids, doc_infos, topic_ids, phrase_infos = map(list, zip(*samples))
    interested_keys = doc_infos[0].keys()
    batched_doc_info, batched_phrase_info = {}, {}
    for k in interested_keys:
        batched_doc_info[k] = torch.stack([ele[k] for ele in doc_infos])
        batched_phrase_info[k] = torch.stack([F.pad(ele[k], (0, 10 - ele[k].shape[0]), "constant", 0) for ele in phrase_infos])
    batched_doc_id_tensor = torch.tensor(doc_ids)
    batched_topic_id_tensor = torch.tensor(topic_ids)  
    return batched_doc_id_tensor, batched_doc_info, batched_topic_id_tensor, batched_phrase_info

"""
    DataLoader (and CollateFn) for Expansion Step
"""

class DocDataLoader(BaseDataLoader):
    def __init__(self, directory, batch_size=16, shuffle=True, validation_split=0.0, num_workers=1, alpha=0.5):
        self.dataset = DocTopicPhraseDataset(directory=directory, batch_type="doc_only", alpha=alpha, raw=False)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=doc_dataset_collate_fn)
        self.n_samples = len(self.dataset)

def doc_dataset_collate_fn(samples):
    doc_ids, doc_infos = map(list, zip(*samples))
    interested_keys = doc_infos[0].keys()
    batched_doc_info = {}
    for k in interested_keys:
        batched_doc_info[k] = torch.stack([ele[k] for ele in doc_infos])
    batched_doc_id_tensor = torch.tensor(doc_ids)
    return batched_doc_id_tensor, batched_doc_info

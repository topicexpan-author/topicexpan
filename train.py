import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.model_zoo as model_zoo
from torchnlp.word_to_vector import GloVe
from parse_config import ConfigParser
from trainer import Trainer, AdamW

SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_for_training', module_data)
    valid_data_loader = data_loader.split_validation()
    
    # build model architecture, then print to console
    topic_hierarchy = data_loader.dataset.topic_hier
    topic_node_feats = data_loader.dataset.topic_node_feats
    topic_mask_feats = data_loader.dataset.topic_mask_feats
    novel_topic_hierarhcy = data_loader.dataset.novel_topic_hier
    model = config.init_obj('arch', module_arch, \
                    topic_hierarchy=topic_hierarchy, \
                    topic_node_feats=topic_node_feats, \
                    topic_mask_feats=topic_mask_feats, \
                    pad_token_id=data_loader.dataset.bert_tokenizer.pad_token_id, \
                    bos_token_id=data_loader.dataset.bert_tokenizer.bos_token_id, \
                    eos_token_id=data_loader.dataset.bert_tokenizer.eos_token_id, \
                    novel_topic_hierarchy=novel_topic_hierarhcy)

    logger.info(model)

    # get function handles of loss and metrics
    criterions = {crt: getattr(module_loss, criterion) for crt, criterion in config['loss'].items()}
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config['optimizer']['args']['weight_decay'],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['optimizer']['args']['lr'], eps=1e-8)

    trainer = Trainer(model, criterions, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader)
                      
    trainer.train()
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='TopicExpan')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, 
                      help='suffix indicating this run (default: None)')
    config = ConfigParser.from_args(args)
    main(config)

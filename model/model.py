import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from itertools import chain
from .model_zoo import *


class TopicExpan(BaseModel):
    """
        Unified Model of TopicExpan
    """
    def __init__(self, pad_token_id, bos_token_id, eos_token_id, **options):
        super(TopicExpan, self).__init__()

        self.doc_encoder = BertDocEncoder(options["model_name"])

        self.phrase_decoder = TransformerPhraseDecoder(
                                    self.doc_encoder.input_embeddings, 
                                    pad_token_id, bos_token_id, eos_token_id,
                                    options["tfm_decoder_num_layers"], 
                                    options["tfm_decoder_num_heads"], 
                                    options["tfm_decoder_max_length"])

        self.topic_hier = options["topic_hierarchy"]
        self.novel_topic_hier = options["novel_topic_hierarchy"]
        self.vid2pid = {vid: pid for vid, pid in enumerate(self.topic_hier)}

        self.topic_encoder = GCNTopicEncoder(
                                    options["topic_hierarchy"], 
                                    options["topic_node_feats"], 
                                    options["topic_mask_feats"],
                                    options["gcn_encoder_num_layers"])

        doc_dim, topic_dim = options["doc_embed_dim"], options["topic_embed_dim"]
        num_topics = options["topic_node_feats"].shape[0]
        assert options["topic_embed_dim"] == options["topic_node_feats"].shape[1]
        
        self.interaction = BilinearInteraction(doc_dim, topic_dim, num_topics=num_topics, bias=False)
        self.linear_combiner = nn.Linear(doc_dim + topic_dim, doc_dim)


    def to_device(self, device):
        self.to(device)
        self.topic_encoder.to_device(device)
        return self

    def forward(self, encoder_input, decoder_input, topic_indices):
        topic_encoder_output = self.topic_encoder.encode()
        doc_encoder_output = self.doc_encoder(encoder_input)
        
        # Part 1 : Topic-Document Similarity Prediction
        doc_tensor = (doc_encoder_output * encoder_input['attention_mask'][:, :, None]).sum(dim=1) 
        doc_tensor /= encoder_input['attention_mask'].sum(dim=1, keepdim=True)
        sim_score = self.interaction(doc_tensor, topic_encoder_output) 
        
        # Part 2 : Topic-conditional Phrase Generation (w/ Teacher Forcing)
        decoder_context = self.context_combiner(topic_encoder_output[topic_indices, :], doc_encoder_output, encoder_input['attention_mask'])
        decoder_output = self.phrase_decoder(decoder_input, decoder_context)
        gen_score = F.log_softmax(decoder_output, dim=-1)

        return sim_score, gen_score

    # Part 1. Topic-Document Similarity Prediction
    def sim(self, encoder_input):
        topic_encoder_output = self.topic_encoder.encode() 
        
        doc_encoder_output = self.doc_encoder(encoder_input)
        doc_tensor = (doc_encoder_output * encoder_input['attention_mask'][:, :, None]).sum(dim=1) 
        doc_tensor /= encoder_input['attention_mask'].sum(dim=1, keepdim=True)
        
        score = self.interaction(doc_tensor, topic_encoder_output) 
        return score

    def inductive_sim(self, encoder_input):
        topic_encoder_output = self.topic_encoder.inductive_encode() 
        topic_encoder_output = torch.stack([topic_encoder_output[pid] for vid, pid in self.vid2pid.items()])

        doc_encoder_output = self.doc_encoder(encoder_input)
        doc_tensor = (doc_encoder_output * encoder_input['attention_mask'][:, :, None]).sum(dim=1) 
        doc_tensor /= encoder_input['attention_mask'].sum(dim=1, keepdim=True)

        score = self.interaction(doc_tensor, topic_encoder_output)
        return score

    # Step 2. Topic-conditional Phrase Generation
    def gen(self, encoder_input, topic_indices):
        topic_encoder_output = self.topic_encoder.encode()[topic_indices, :]
        
        doc_encoder_output = self.doc_encoder(encoder_input)
        doc_encoder_mask = encoder_input['attention_mask']

        decoder_context = self.context_combiner(topic_encoder_output, doc_encoder_output, doc_encoder_mask)
        output_ids = self.phrase_decoder.generate(decoder_context)
        return output_ids

    def inductive_gen(self, encoder_input, topic_indices):
        topic_encoder_output = self.topic_encoder.inductive_encode() 
        topic_encoder_output = torch.stack([topic_encoder_output[pid] for vid, pid in self.vid2pid.items()])
        topic_encoder_output = topic_encoder_output[topic_indices, :]

        doc_encoder_output = self.doc_encoder(encoder_input)
        doc_encoder_mask = encoder_input['attention_mask']

        decoder_context = self.context_combiner(topic_encoder_output, doc_encoder_output, doc_encoder_mask)
        output_ids = self.phrase_decoder.generate(decoder_context)
        return output_ids

    # generation with Teacher Forcing
    def gen_with_tf(self, encoder_input, decoder_input, topic_indices):
        topic_encoder_output = self.topic_encoder.encode()[topic_indices, :]
        doc_encoder_output = self.doc_encoder(encoder_input)
        doc_encoder_mask = encoder_input['attention_mask']

        decoder_context = self.context_combiner(topic_encoder_output, doc_encoder_output, doc_encoder_mask)
        decoder_output = self.phrase_decoder(decoder_input, decoder_context)
        gen_score = F.log_softmax(decoder_output, dim=-1)
        return gen_score

    def context_combiner(self, topic_context, doc_context, doc_mask):        
        scores = self.interaction.compute_attn_scores(doc_context, topic_context)
        scores = (torch.exp(scores) + 1e-6) * doc_mask
        scores /= scores.sum(dim=1, keepdim=True)

        context = (doc_context * scores.unsqueeze(dim=2))
        return context
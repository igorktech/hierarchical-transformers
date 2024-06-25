import argparse

import torch
import copy
import warnings
from data import DATA_DIR
from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.hat import HATForMaskedLM, HATConfig, HATTokenizer
warnings.filterwarnings("ignore")

LAYOUTS = {
    's1': 'SD|SD|SD|SD|SD|SD',
    's2': 'S|SD|D|S|SD|D|S|SD|D',
    'p1': 'S|SD|S|SD|S|SD|S|SD',
    'p2': 'S|S|SD|S|S|SD|S|S|SD',
    'p2_2': 'S|S|SD|S|S|SD',
    'e1': 'SD|SD|SD|S|S|S|S|S|S',
    'e2': 'S|SD|D|S|SD|D|S|S|S|S',
    'l1': 'S|S|S|S|S|S|SD|SD|SD',
    'l2': 'S|S|S|S|S|SD|D|S|SD|D',
    'b1': 'S|S|SD|D|S|SD|D|S|S|S',
    'b2': 'S|S|SD|SD|SD|S|S|S|S',
    'f12': 'S|S|S|S|S|S|S|S|S|S|S|S',
    'f8': 'S|S|S|S|S|S|S|S',
    'f6': 'S|S|S|S|S|S',
}


def convert_bert_to_htf():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--warmup_strategy', default='grouped', choices=['linear', 'grouped', 'random', 'embeds-only', 'none'],
                        help='linear: S|D encoders are warm-started independently (one-by-one)'
                             'grouped: pairs of S|D are warm-started with weights from the very same level'
                             'random: D encoders are not warm-started'
                             'embeds-only: No warm-starting, except embeddings'
                             'none: No warm-starting')
    parser.add_argument('--layout', default='s1', choices=['s1', 's2', 'p1', 'p2','p2_2', 'e1', 'e2',
                                                           'l1', 'l2', 'b1', 'b2', 'f12', 'f8', 'f6'],
                        help='S|D encoders layout')
    parser.add_argument('--max_sentences', default=8)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_sentence_length', default=128)

    config = parser.parse_args()
    MAX_SENTENCES = int(config.max_sentences)
    MAX_SENTENCE_LENGTH = int(config.max_sentence_length)
    ENCODER_LAYOUT = {}
    for idx, block_pattern in enumerate(LAYOUTS[config.layout].split('|')):
        ENCODER_LAYOUT[str(idx)] = {"sentence_encoder": True if 'S' in block_pattern else False,
                                    "document_encoder": True if 'D' in block_pattern else False}

    NUM_HIDDEN_LAYERS = len(ENCODER_LAYOUT.keys())
    BERT_LAYERS = NUM_HIDDEN_LAYERS if config.warmup_strategy != 'linear' else NUM_HIDDEN_LAYERS*2
    BERT_LAYERS = BERT_LAYERS + 1 if BERT_LAYERS % 2 else BERT_LAYERS
    if config.model_name is None:
        BERT_CHECKPOINT = f'google/bert_uncased_L-{str(BERT_LAYERS)}_H-256_A-4'
    else:
        BERT_CHECKPOINT = config.model_name

    # load pre-trained bert model and tokenizer
    bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_SENTENCE_LENGTH * MAX_SENTENCES)

    # load dummy config and change specifications
    bert_config = bert_model.config
    htf_config = HATConfig.from_pretrained(f'{DATA_DIR}/hat')
    # Text length parameters
    htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
    htf_config.max_sentences = MAX_SENTENCES
    htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH
    htf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    htf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Transformer parameters
    try:
        htf_config.hidden_size = bert_config.hidden_size
    except:
        htf_config.hidden_size = bert_config.dim
    try:
        htf_config.intermediate_size = bert_config.intermediate_size
    except:
        htf_config.intermediate_size = bert_config.hidden_dim
    try:
        htf_config.num_attention_heads = bert_config.num_attention_heads
    except:
        htf_config.num_attention_heads = bert_config.n_heads
    try:
        htf_config.hidden_act = bert_config.hidden_act
    except:
        htf_config.hidden_act = bert_config.activation
    htf_config.encoder_layout = ENCODER_LAYOUT
    # Vocabulary parameters
    htf_config.vocab_size = bert_config.vocab_size
    htf_config.pad_token_id = bert_config.pad_token_id
    htf_config.bos_token_id = bert_config.bos_token_id
    htf_config.eos_token_id = bert_config.eos_token_id
    try:
        htf_config.type_vocab_size = bert_config.type_vocab_size
    except:
        htf_config.type_vocab_size = 2

    # load dummy hi-transformer model
    htf_model = HATForMaskedLM.from_config(htf_config)

    if config.warmup_strategy != 'none':
        # copy embeddings
        htf_model.hi_transformer.embeddings.position_embeddings.weight.data[0] = torch.zeros((htf_config.hidden_size,))
        if bert_config.model_type == 'bert':
            htf_model.hi_transformer.embeddings.position_embeddings.weight.data[1:] = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_SENTENCE_LENGTH+htf_config.pad_token_id+1]
            htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
            htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
            htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())
        if bert_config.model_type == 'distilbert':
            htf_model.hi_transformer.embeddings.position_embeddings.weight.data[1:] = bert_model.distilbert.embeddings.position_embeddings.weight[1:MAX_SENTENCE_LENGTH+htf_config.pad_token_id+1]
            htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(bert_model.distilbert.embeddings.word_embeddings.state_dict())
            # htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(bert_model.distilbert.embeddings.token_type_embeddings.state_dict())
            htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(bert_model.distilbert.embeddings.LayerNorm.state_dict())

        if config.warmup_strategy != 'embeds-only':
            # copy transformer layers
            if config.warmup_strategy != 'linear':
                for idx in range(NUM_HIDDEN_LAYERS):
                    if htf_model.config.encoder_layout[str(idx)]['sentence_encoder']:
                        if bert_config.model_type == 'bert':
                            htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(bert_model.bert.encoder.layer[idx].state_dict())
                        if bert_config.model_type == 'distilbert':
                            htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(bert_model.distilbert.transformer.layer[idx].state_dict())
                    if htf_model.config.encoder_layout[str(idx)]['document_encoder']:
                        if config.warmup_strategy == 'grouped':
                            if bert_config.model_type == 'bert':
                                htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(bert_model.bert.encoder.layer[idx].state_dict())
                            if bert_config.model_type == 'distilbert':
                                htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(bert_model.distilbert.transformer.layer[idx].state_dict())
                        if bert_config.model_type == 'bert':
                                htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]
                        if bert_config.model_type == 'distilbert':
                            htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = bert_model.distilbert.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]
            else:
                for idx, l_idx in enumerate(range(0, NUM_HIDDEN_LAYERS*2, 2)):
                    if htf_model.config.encoder_layout[str(idx)]['sentence_encoder']:
                        htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(bert_model.bert.encoder.layer[l_idx].state_dict())
                    if htf_model.config.encoder_layout[str(idx)]['document_encoder']:
                        htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(bert_model.bert.encoder.layer[l_idx+1].state_dict())
                        htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]

        # copy lm_head
        htf_model.lm_head.dense.load_state_dict(bert_model.cls.predictions.transform.dense.state_dict())
        htf_model.lm_head.layer_norm.load_state_dict(bert_model.cls.predictions.transform.LayerNorm.state_dict())
        htf_model.lm_head.decoder.load_state_dict(bert_model.cls.predictions.decoder.state_dict())
        htf_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)

    # save model
    htf_model.save_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')

    # save tokenizer
    tokenizer.save_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')

    # re-load model
    htf_model = HATForMaskedLM.from_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')
    htf_tokenizer = HATTokenizer.from_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')
    print(f'HAT model with layout {config.layout} and warm-up strategy {config.warmup_strategy} is ready to run!')


if __name__ == '__main__':
    convert_bert_to_htf()

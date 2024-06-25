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
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_strategy', default='grouped', choices=['linear', 'grouped', 'random', 'embeds-only', 'none'])
    parser.add_argument('--layout', default='s1', choices=['s1', 's2', 'p1', 'p2','p2_2', 'e1', 'e2', 'l1', 'l2', 'b1', 'b2', 'f12', 'f8', 'f6'])
    parser.add_argument('--max_sentences', default=8)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_sentence_length', default=64)

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

    bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_SENTENCE_LENGTH * MAX_SENTENCES)

    bert_config = bert_model.config
    htf_config = HATConfig.from_pretrained(f'{DATA_DIR}/hat')
    htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
    htf_config.max_sentence_size = MAX_SENTENCE_LENGTH
    htf_config.max_sentences = MAX_SENTENCES
    htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH
    htf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    htf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    htf_config.hidden_size = bert_config.hidden_size if hasattr(bert_config, 'hidden_size') else bert_config.dim
    htf_config.intermediate_size = bert_config.intermediate_size if hasattr(bert_config, 'intermediate_size') else bert_config.hidden_dim
    htf_config.num_attention_heads = bert_config.num_attention_heads if hasattr(bert_config, 'num_attention_heads') else bert_config.n_heads
    htf_config.hidden_act = bert_config.hidden_act if hasattr(bert_config, 'hidden_act') else bert_config.activation
    htf_config.encoder_layout = ENCODER_LAYOUT
    htf_config.vocab_size = bert_config.vocab_size
    htf_config.pad_token_id = bert_config.pad_token_id
    htf_config.bos_token_id = bert_config.bos_token_id
    htf_config.eos_token_id = bert_config.eos_token_id
    htf_config.type_vocab_size = bert_config.type_vocab_size if hasattr(bert_config, 'type_vocab_size') else 2

    htf_model = HATForMaskedLM.from_config(htf_config)

    if config.warmup_strategy != 'none':
        htf_model.hi_transformer.embeddings.position_embeddings.weight.data[0] = torch.zeros((htf_config.hidden_size,))
        if bert_config.model_type == 'bert':
            htf_model.hi_transformer.embeddings.position_embeddings.weight.data[1:] = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_SENTENCE_LENGTH+htf_config.pad_token_id+1]
            htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
            htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
            htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())
        elif bert_config.model_type == 'distilbert':
            htf_model.hi_transformer.embeddings.position_embeddings.weight.data[1:] = bert_model.distilbert.embeddings.position_embeddings.weight[1:MAX_SENTENCE_LENGTH+htf_config.pad_token_id+1]
            htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(bert_model.distilbert.embeddings.word_embeddings.state_dict())
            htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(bert_model.distilbert.embeddings.LayerNorm.state_dict())

        if config.warmup_strategy != 'embeds-only':
            for idx in range(NUM_HIDDEN_LAYERS):
                if htf_model.config.encoder_layout[str(idx)]['sentence_encoder']:
                    if bert_config.model_type == 'bert':
                        htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(bert_model.bert.encoder.layer[idx].state_dict())
                    elif bert_config.model_type == 'distilbert':
                        distil_state = bert_model.distilbert.transformer.layer[idx].state_dict()
                        hat_state = {}
                        
                        # Attention mappings
                        hat_state['attention.self.query.weight'] = distil_state['attention.q_lin.weight']
                        hat_state['attention.self.query.bias'] = distil_state['attention.q_lin.bias']
                        hat_state['attention.self.key.weight'] = distil_state['attention.k_lin.weight']
                        hat_state['attention.self.key.bias'] = distil_state['attention.k_lin.bias']
                        hat_state['attention.self.value.weight'] = distil_state['attention.v_lin.weight']
                        hat_state['attention.self.value.bias'] = distil_state['attention.v_lin.bias']
                        hat_state['attention.output.dense.weight'] = distil_state['attention.out_lin.weight']
                        hat_state['attention.output.dense.bias'] = distil_state['attention.out_lin.bias']
                        
                        # LayerNorm mappings
                        hat_state['attention.output.LayerNorm.weight'] = distil_state['sa_layer_norm.weight']
                        hat_state['attention.output.LayerNorm.bias'] = distil_state['sa_layer_norm.bias']
                        hat_state['output.LayerNorm.weight'] = distil_state['output_layer_norm.weight']
                        hat_state['output.LayerNorm.bias'] = distil_state['output_layer_norm.bias']
                        
                        # FFN mappings
                        hat_state['intermediate.dense.weight'] = distil_state['ffn.lin1.weight']
                        hat_state['intermediate.dense.bias'] = distil_state['ffn.lin1.bias']
                        hat_state['output.dense.weight'] = distil_state['ffn.lin2.weight']
                        hat_state['output.dense.bias'] = distil_state['ffn.lin2.bias']
                        
                        htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(hat_state)
                
                if htf_model.config.encoder_layout[str(idx)]['document_encoder']:
                    if config.warmup_strategy == 'grouped':
                        if bert_config.model_type == 'bert':
                            htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(bert_model.bert.encoder.layer[idx].state_dict())
                        elif bert_config.model_type == 'distilbert':
                            distil_state = bert_model.distilbert.transformer.layer[idx].state_dict()
                            hat_state = {}
                            
                            # Attention mappings
                            hat_state['attention.self.query.weight'] = distil_state['attention.q_lin.weight']
                            hat_state['attention.self.query.bias'] = distil_state['attention.q_lin.bias']
                            hat_state['attention.self.key.weight'] = distil_state['attention.k_lin.weight']
                            hat_state['attention.self.key.bias'] = distil_state['attention.k_lin.bias']
                            hat_state['attention.self.value.weight'] = distil_state['attention.v_lin.weight']
                            hat_state['attention.self.value.bias'] = distil_state['attention.v_lin.bias']
                            hat_state['attention.output.dense.weight'] = distil_state['attention.out_lin.weight']
                            hat_state['attention.output.dense.bias'] = distil_state['attention.out_lin.bias']
                            
                            # LayerNorm mappings
                            hat_state['attention.output.LayerNorm.weight'] = distil_state['sa_layer_norm.weight']
                            hat_state['attention.output.LayerNorm.bias'] = distil_state['sa_layer_norm.bias']
                            hat_state['output.LayerNorm.weight'] = distil_state['output_layer_norm.weight']
                            hat_state['output.LayerNorm.bias'] = distil_state['output_layer_norm.bias']
                            
                            # FFN mappings
                            hat_state['intermediate.dense.weight'] = distil_state['ffn.lin1.weight']
                            hat_state['intermediate.dense.bias'] = distil_state['ffn.lin1.bias']
                            hat_state['output.dense.weight'] = distil_state['ffn.lin2.weight']
                            hat_state['output.dense.bias'] = distil_state['ffn.lin2.bias']
                            
                            htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(hat_state)
                    
                    if bert_config.model_type == 'bert':
                        htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]
                    elif bert_config.model_type == 'distilbert':
                        htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = bert_model.distilbert.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]

        # copy lm_head
        if bert_config.model_type == 'bert':
            htf_model.lm_head.dense.load_state_dict(bert_model.cls.predictions.transform.dense.state_dict())
            htf_model.lm_head.layer_norm.load_state_dict(bert_model.cls.predictions.transform.LayerNorm.state_dict())
            htf_model.lm_head.decoder.load_state_dict(bert_model.cls.predictions.decoder.state_dict())
            htf_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)
        elif bert_config.model_type == 'distilbert':
            htf_model.lm_head.dense.load_state_dict(bert_model.vocab_transform.state_dict())
            htf_model.lm_head.layer_norm.load_state_dict(bert_model.vocab_layer_norm.state_dict())
            htf_model.lm_head.decoder.load_state_dict(bert_model.vocab_projector.state_dict())
            # DistilBERT doesn't have a separate bias, it's included in the vocab_projector
            htf_model.lm_head.bias = copy.deepcopy(bert_model.vocab_projector.bias)
    # save model
    htf_model.save_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}',safe_serialization=False)

    # save tokenizer
    tokenizer.save_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')

    # re-load model
    htf_model = HATForMaskedLM.from_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')
    htf_tokenizer = HATTokenizer.from_pretrained(f'{DATA_DIR}/PLMs/hat-{config.layout}-{config.warmup_strategy}')
    print(f'HAT model with layout {config.layout} and warm-up strategy {config.warmup_strategy} is ready to run!')

if __name__ == '__main__':
    convert_bert_to_htf()
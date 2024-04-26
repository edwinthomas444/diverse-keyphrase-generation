import torch
from models.modelling import BertForTokenClassification
from models.modelling_encoder_decoder import EncoderDecoderModel
# from transformers import BertGenerationDecoder
import os
from transformers import EncoderDecoderConfig  # , EncoderDecoderModel
from models.modelling_bert_generation import BertGenerationDecoder
# present key phrase base models


def pkp_base_loader(
        config,
        **kwargs
):
    sd = None
    if config['model']['encoder_state_dict']:
        sd = torch.load(os.path.join(
            config['project_path'], config['model']['encoder_state_dict']))
    model = BertForTokenClassification.from_pretrained(config['model']['name'],
                                                       state_dict=sd,
                                                       type_vocab_size=config['model']['type_vocab_size'],
                                                       vocab_size=config['model']['tokenizer_vocab_size'],
                                                       num_labels=config['model']['num_labels'])
    
    ep_offset = 0 #os.path.basename(config['model']['encoder_state_dict']).split('.')[1]
    model.config.epoch_offset = ep_offset

    return model

# absent key phrase
# base models


def akp_base_loader(
        config,
        tokenizer,
        **kwargs
):

    
    model = None
    # create new model from huggingface checkpoint weights
    # supported checkpoints
    hugging_face_chkpts = ['bert-base-uncased']
    
    if 'external_seq2set' in kwargs:
        decoder_kwargs = kwargs
    else:
        decoder_kwargs = {
            'external_seq2set':False,
            'external_max_kps': None,
            'external_max_kp_len': None,
            'external_diversity_heads': False,
            'external_precision_heads': False,
            'external_precision_attn_layers': False}

    if config['model']['encoder_decoder_state_dict']:
        # try to load from checkpoint path directly (needs config also saved in directory of checkpoint)
        enc_dec_checkpoint_path = os.path.join(
            config['project_path'], config['model']['encoder_decoder_state_dict'])
        enc_dec_config_path = os.path.join(config['project_path'], os.path.dirname(
            config['model']['encoder_decoder_state_dict']), 'config.json')
        # Assumption: model load name is in the format model.x.bin where x is the latest epoch
        # for which model was trained
        ep_offset = os.path.basename(config['model']['encoder_decoder_state_dict']).split('.')[1]

        if os.path.isfile(enc_dec_config_path):
            enc_dec_cf = EncoderDecoderConfig.from_pretrained(
                enc_dec_config_path)
            
            # update the model config to include precision heads (in-case we want to resume from an ablation point)
            mod = enc_dec_cf.decoder
            mod.update({'precision_heads': decoder_kwargs['external_precision_heads']})
            mod.update({'precision_attn_layers': decoder_kwargs['external_precision_attn_layers']})
            config.update({'decoder': mod})

            print('Encoder Decoder config: ', enc_dec_cf)
        else:
            raise Exception('encoder decoder model checkpoint path does not exist')

        model = EncoderDecoderModel.from_pretrained(
            enc_dec_checkpoint_path, config=enc_dec_cf)
        # set epoch offset after loading
        model.config.epoch_offset = int(ep_offset) + 1

    elif config['model']['encoder_state_dict'] and config['model']['decoder_name'] in hugging_face_chkpts:
        enc_checkpoint_path = os.path.join(
            config['project_path'], os.path.dirname(config['model']['encoder_state_dict']))

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            enc_checkpoint_path, config['model']['decoder_name'], **decoder_kwargs)
        model.config.epoch_offset = 0
        
    elif config['model']['encoder_name'] in hugging_face_chkpts and \
            config['model']['decoder_name'] in hugging_face_chkpts:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            config['model']['encoder_name'], config['model']['decoder_name'], **decoder_kwargs)
        model.config.epoch_offset = 0

    if model is not None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_len_d = config['tokenizer']['max_len_d']
        # for decoder generate step
        model.config.max_length = config['run_params']['max_length']
        model.config.min_length = config['run_params']['min_length']
        model.config.no_repeat_ngram_size = config['run_params']['no_repeat_ngram_size']
        model.config.early_stopping = config['run_params']['early_stopping']
        model.config.length_penalty = config['run_params']['length_penalty']
        model.config.num_beams = config['run_params']['num_beams']
        model.config.num_return_sequences = config['run_params']['num_return_sequences']
        model.config.seq2set = False
        # encoder mlm is not used for seq2seq
        model.config.encoder_mlm = False
        return model
    else:
        raise Exception('Cannot load model with given name and configs')


# seq2set loader


def akp_seq2set_loader(config,
                       tokenizer,
                       **kwargs
                       ):
    
    max_kps = config['model']['max_kps']
    max_kp_len = config['model']['max_kp_len']
    decoder_kwargs = {
        'external_seq2set': True,
        'external_max_kps': max_kps,
        'external_max_kp_len': max_kp_len,
        'external_diversity_heads': config['model']['diversity_heads'],
        'external_precision_heads': config['model']['precision_heads'],
        'external_precision_attn_layers': config['model']['precision_attn_layers'],
        'decoder_type_vocab_size': config['model']['decoder_type_vocab_size']
    }

    model = akp_base_loader(config=config, tokenizer=tokenizer, **decoder_kwargs)
    if config['run_params']['freeze_encoder']:
        for param_name, param in model.named_parameters():
            if param_name.startswith('encoder'):
                param.requires_grad = False

    # update the model config to include max_kps and max_kp_len for seq2set
    model.config.max_kps = max_kps
    model.config.max_kp_len = max_kp_len
    model.config.seq2set = True
    model.config.cross_unit_attention = config['run_params']['cross_unit_attention']
    model.config.hungarian_assign = config['run_params']['hungarian_assign']
    # configure generate max_length based on max_kps and max_kp_len
    model.config.max_length = model.config.max_kp_len #model.config.max_kps*model.config.max_kp_len
    model.config.early_stopping = False
    model.config.min_length = 1#-1
    # generate till max_length, min_length is deactived
    # eos token needs to be replaced with some unused token
    # for independant decoding
    # model.config.eos_token_id = [tokenizer.sep_token_id, tokenizer.pad_token_id] #tokenizer.sep_token_id #tokenizer.convert_tokens_to_ids('[MASK]')
    model.config.eos_token_id = [tokenizer.sep_token_id]
    # remove penalties on n-gram repetitions as within a unit, the special tokens can repeat until start of next token
    model.config.no_repeat_ngram_size = None
    # also remove length penalty
    model.config.length_penalty = 0

    # whether to pass all encoder hidden states or only those
    # corresponding to positions of the extractive key-phrases
    model.config.filter_extkp_hidden_states = config['run_params']['filter_extkp_hidden_states']
    model.config.max_len_e = config['tokenizer']['max_len']
    model.config.encoder_mlm = config['model']['encoder_mlm']
    if model.config.encoder_mlm:
        # innitialize mlm head
        model.add_mlm_head()

    # model.config.to_json_file('/home/thomased/work/codebase/coopsummer2023/Project/checkpoints/phase2_exp/abstractive_models/test11_diversity_nohungarian/config1.json')
    return model


# proposed model loader
def akp_emb2set_loader(config,
                       **kwargs):
    sd = None
    if config['model']['decoder_state_dict']:
        sd = torch.load(os.path.join(
            config['project_path'], config['model']['decoder_state_dict']))

    model = BertGenerationDecoder.from_pretrained(config['model']['name'],
                                                  state_dict=sd,
                                                  type_vocab_size=config['model']['type_vocab_size'])
    model.config.max_kps = config['model']['max_kps']
    model.config.max_kp_len = config['model']['max_kp_len']

    # using as decoder
    model.config.vanilla_nar = config['run_params']['vanilla_nar']

    model.config.is_decoder = True if model.config.vanilla_nar else False
    # whether to use hidden states of encoder as input
    # or corresponding extractive token ids as input
    model.config.input_embeds = config['run_params']['input_embeds']

    # adding RNN head to model local dependancies for NAR type architectures
    model.config.rnn_head = config['model']['rnn_head']
    if model.config.rnn_head:
        model.add_tiny_rnn_head(max_units=model.config.max_kps)
        
    return model

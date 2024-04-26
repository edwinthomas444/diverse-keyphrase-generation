import time
import torch
import copy
import numpy as np
from models.model_loaders import pkp_base_loader
from dataset.utils import get_goldkp, get_goldakp
import sys
sys.path.append('./')
# from utils import get_goldkp, get_goldakp


class PKPBasePipeline:
    def __init__(self,
                 data_tokenizer,
                 model_tokenizer,
                 truncate_config,
                 max_len,
                 label_map,
                 **kwargs):
        self.data_tokenizer = data_tokenizer
        self.model_tokenizer = model_tokenizer
        self.truncate_config = truncate_config
        self.max_len = max_len
        self.label_map = label_map

    def __call__(self, data_point, is_train):
        # preprocess the data (tokens or labels in list)
        doc = self.data_tokenizer.tokenize(
            data_point['doc'].strip(),
            truncate=self.truncate_config['len_a'])

        labels = self.data_tokenizer.tokenize(
            data_point['lab'].strip(),
            truncate=self.truncate_config['len_a'])
        # add special tokens
        doc_list = ['[CLS]']+doc+['[SEP]']
        label_list = ['O']+labels+['O']
        assert len(doc_list) == len(
            label_list), f'doc_list length {len(doc_list)} != label list length {len(label_list)}'
        # dict with keys as keyphrases and values as positions (start, end)
        gold_kps = get_goldkp(doc_list, label_list)
        doc_tokens = [data_point['doc']]

        # convert tokens to input_ids for model
        input_ids = self.model_tokenizer.convert_tokens_to_ids(doc_list)
        assert len(input_ids) == len(
            label_list), f'# doc tokens {len(input_ids)} not equal to # of labels {len(label_list)}'
        assert (self.truncate_config['len_a']-2) <= self.max_len, \
            f'''Maximum seq_a length {self.truncate_config['len_a']} 
                cannot be fit within max total len {self.max_len}'''

        # pad to max length
        pad_length = self.max_len - len(label_list)
        attention_mask = [0] + [1]*len(labels) + [0]*(pad_length+1)
        token_type_ids = [1]*self.max_len

        label_ids = None
        if self.label_map:
            label_ids = [self.label_map[lab]
                         for lab in label_list] + [self.label_map['O']]*pad_length

        input_ids = input_ids + [0]*pad_length

        ds = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'label_ids': label_ids,
            'attention_mask': attention_mask,
            'gold_kps': gold_kps,
            'doc': doc_tokens
        }

        return ds


# Absent KP model data pipeline definition
class AKPSeq2SeqPipeline:
    def __init__(self,
                 data_tokenizer,
                 model_tokenizer,
                 truncate_config,
                 max_len,
                 max_len_d,
                 **kwargs):
        self.data_tokenizer = data_tokenizer
        self.model_tokenizer = model_tokenizer
        self.truncate_config = truncate_config
        self.max_len_encoder = max_len
        self.max_len_decoder = max_len_d

    def __call__(self, data_point, is_train):
        # preprocess the data (tokens or labels in list)
        doc = self.data_tokenizer.tokenize(
            data_point['doc'].strip(),
            truncate=self.truncate_config['len_a'])

        # semi colon seperates each key phrase, white space tokenization for sequence of AKP
        target = self.data_tokenizer.tokenize(
            data_point['tgt'].strip(),
            truncate=self.truncate_config['len_b'])

        # get list of target gold Absent KPs
        target_gold_kps = get_goldakp(target=data_point['tgt'].strip())

        # add special tokens
        doc_list = ['[CLS]']+doc+['[SEP]']
        target_list = ['[CLS]']+target+['[SEP]']

        # get input document tokens
        doc_tokens = [data_point['doc']]

        assert (self.truncate_config['len_a']-2) <= self.max_len_encoder, \
            f'''Maximum seq_a length {self.truncate_config['len_a']} 
                cannot be fit within max total len {self.max_len_encoder}'''
        assert (self.truncate_config['len_b']-2) <= self.max_len_decoder, \
            f'''Maximum seq_b length {self.truncate_config['len_b']} 
                cannot be fit within max total len {self.max_len_decoder}'''

        # pad to max length
        encoder_pad_length = self.max_len_encoder - len(doc_list)
        decoder_pad_length = self.max_len_decoder - len(target_list)

        # convert tokens to input_ids for model
        encoder_input_ids = self.model_tokenizer.convert_tokens_to_ids(
            doc_list) + [0]*encoder_pad_length
        encoder_attention_mask = [
            1]*len(doc_list) + [0]*encoder_pad_length

        # default mlm mask added (not used)
        encoder_mlm_labels_mask = [0]*self.max_len_encoder

        # decoder target ids and input ids are same
        # the decoder model uses predictions for input_ids[:-1] and target[1:] as target and
        # for computation of loss in forward
        # -100 added instead of PAD token id as model internally considers it as [PAD] token
        # as both decoder input and target ids are provided, PAD not replaced with -100
        # only if decoder input ids is not given and prepared from labels then labels
        # should have -100 for pad tokens
        decoder_target_ids = self.model_tokenizer.convert_tokens_to_ids(
            target_list) + [0]*decoder_pad_length
        decoder_input_ids = self.model_tokenizer.convert_tokens_to_ids(
            target_list) + [0]*decoder_pad_length
        # get causal decoder attention mask
        # uni-directional attention from left to right
        # achieved through 2d causal attention mask
        decoder_seq_len = len(decoder_input_ids)
        # decoder_attention_mask = np.tril(
        #     np.ones((decoder_seq_len, decoder_seq_len)))

        # MOD: exclude pad tokens in 2d mask along row and col dimensions
        actual_target_length = len(target_list)
        decoder_attention_mask = np.zeros((decoder_seq_len, decoder_seq_len))
        causal_mask = np.tril(np.ones((decoder_seq_len, decoder_seq_len)))
        decoder_attention_mask[:actual_target_length,
                               :actual_target_length] = causal_mask[:actual_target_length, :actual_target_length]

        loss_weight_mask = [1]*self.max_len_decoder
        
        ds = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'cross_attention_mask': encoder_attention_mask,
            'encoder_mlm_labels_mask': encoder_mlm_labels_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': decoder_target_ids,
            'doc': doc_tokens,
            'gold_akp': target_gold_kps,
            'loss_weight_mask': loss_weight_mask
        }

        return ds

# new pipeline for treating key phrases as units
# k units and m tokens in total per unit
# bs, k, m
# decoder_input_ids: change the data tokenizer

# define pipeline for loading pkp getting predicted text


class PKPInferencePipeline:
    def __init__(self,
                 data_tokenizer,
                 model_tokenizer,
                 trunate_config,
                 max_len,
                 label_map,
                 inv_label_map,
                 model_load_config,
                 **kwargs):
        self.data_tokenizer = data_tokenizer
        self.model_tokenizer = model_tokenizer
        self.truncate_config = trunate_config
        self.max_len_encoder = max_len
        self.label_map = label_map
        self.inv_label_map = inv_label_map
        self.model_load_config = model_load_config

        # innitialize encoder pipeline
        self.encoder_pipeline = PKPBasePipeline(
            data_tokenizer=self.data_tokenizer,
            model_tokenizer=self.model_tokenizer,
            truncate_config=self.truncate_config,
            max_len=self.max_len_encoder,
            label_map=None
        )

        # define model for obtaining ext kps
        self.enc_device = torch.device('cuda')
        self.encoder_model = self._load_encoder(config=self.model_load_config)
        self.encoder_model.to(self.enc_device)

    def _load_encoder(self, config):
        model = pkp_base_loader(config=config)
        return model

    def __call__(self, data_point, is_train):
        pre_inp = self.encoder_pipeline(data_point=data_point, is_train=False)
        self.encoder_model.eval()

        with torch.no_grad():
            enc_inp_ids = torch.tensor(
                pre_inp['input_ids']).unsqueeze(0).to(self.enc_device)
            enc_tok_ids = torch.tensor(
                pre_inp['token_type_ids']).unsqueeze(0).to(self.enc_device)
            enc_attn_mask = torch.tensor(
                pre_inp['attention_mask']).unsqueeze(0).to(self.enc_device)
            # doc_text = pre_inp['doc'][0]

            model_out = self.encoder_model(
                input_ids=enc_inp_ids,
                token_type_ids=enc_tok_ids,
                attention_mask=enc_attn_mask,
                labels=None,
                output_hidden_states=True,
                return_dict=True
            )
            # get last hidden state
            # all_hidden_states = model_out['hidden_states']
            # hidden_state = all_hidden_states[-1]
            logits = model_out['logits']
            # ignoring [CLS] token
            doc_prob = (logits.detach().cpu().numpy())[0]
            doc_pred = ((torch.argmax(logits, dim=2)
                         ).detach().cpu().numpy())[0]
            # ignore 'O' for first [CLS] and [PAD] tokens
            doc_pred_labels = [self.inv_label_map[x] for x in doc_pred][1:-1]

        return doc_pred_labels, doc_prob


class AKPSeq2SetPipeline:
    def __init__(self,
                 data_tokenizer,
                 kp_tokenizer,
                 model_tokenizer,
                 truncate_config,
                 decoder_output_config,
                 max_len,
                 inv_label_map,
                 label_map,
                 model_load_config,
                 cross_unit_attention,
                 ignore_pad_attention,
                 add_phi_tokens,
                 filter_encoder_hidden_states,
                 gt_setting,
                 use_pseudo_labels,
                 encoder_masking,
                 control_codes,
                 **kwargs):
        self.data_tokenizer = data_tokenizer
        self.kp_tokenizer = kp_tokenizer
        self.model_tokenizer = model_tokenizer
        self.truncate_config = truncate_config
        self.max_len_encoder = max_len
        self.decoder_output_config = decoder_output_config
        self.cross_unit_attention = cross_unit_attention
        self.ignore_pad_attention = ignore_pad_attention
        self.add_phi_tokens = add_phi_tokens
        self.filter_encoder_hidden_states = filter_encoder_hidden_states
        self.gt_setting = gt_setting
        self.inv_label_map = inv_label_map
        self.label_map = label_map
        self.model_load_config = model_load_config
        self.use_pseudo_labels = use_pseudo_labels
        self.encoder_masking = encoder_masking
        self.control_codes = control_codes
        if self.filter_encoder_hidden_states and (not self.gt_setting or self.use_pseudo_labels):
            self.pkp_inference_pipeline = PKPInferencePipeline(data_tokenizer=self.data_tokenizer,
                                                               model_tokenizer=self.model_tokenizer,
                                                               trunate_config=self.truncate_config,
                                                               max_len=self.max_len_encoder,
                                                               label_map=self.label_map,
                                                               inv_label_map=self.inv_label_map,
                                                               model_load_config=self.model_load_config)

    def __call__(self, data_point, is_train):
        # preprocess the data (tokens or labels in list)
        doc = self.data_tokenizer.tokenize(
            data_point['doc'].strip(),
            truncate=self.truncate_config['len_a'])

        # get unit kps (truncated by max_kps)
        target_kps = self.kp_tokenizer.tokenize(
            data_point['tgt'].strip(),
            max_kps=self.decoder_output_config['max_kps'])

        # for target_kp unit list by padding to max unit length
        decoder_target_tokens, decoder_input_tokens = [], []
        decoder_causal_mask = []  # 2d mask to support independant decoding of outputs

        # params
        cross_unit_attention = self.cross_unit_attention
        max_kp_len = self.decoder_output_config['max_kp_len']
        max_kps = self.decoder_output_config['max_kps']
        total_seq_len = max_kp_len*max_kps

        target_kps_olen = len(target_kps)
        none_dp = target_kps_olen == 1 and target_kps[0] == 'none'

        if self.add_phi_tokens:
            # all units without AKP are represented as [CLS]+[none]+[SEP]+[PAD]..[PAD], where none is the phi token
            target_kps += ['none']*(max_kps-len(target_kps))

        # print('\ntarget kps: ', target_kps)

        for kp_i, kp in enumerate(target_kps):
            kp_tokens = kp.split(" ")[:max_kp_len-2]
            # -1 for [CLS] and [SEP] each
            pad_length = max_kp_len - len(kp_tokens) - 2

            # for pad units the terminating eos is PAD so that SEP doesnt get extra loss weight
            # loss is weighted down only on none tokens, for pad units sep should not have higher weight
            # so sep removed in pad units
            curr_eos = '[PAD]' if kp=='none' else '[SEP]'

            curr_decoder_tokens = ['[CLS]'] + \
                kp_tokens + [curr_eos] + ['[PAD]']*pad_length

            decoder_target_tokens += copy.deepcopy(curr_decoder_tokens)
            decoder_input_tokens += copy.deepcopy(curr_decoder_tokens)

            # compute 2d causal mask
            curr_len = len(curr_decoder_tokens)
            for i in range(curr_len):
                # first_col = [decoder_causal_mask[row_i][0]
                #              for row_i in range(len(decoder_causal_mask))]

                fill = 1 if (
                    curr_decoder_tokens[i] != '[PAD]' or not self.ignore_pad_attention) else 0
                # if fill is 0 and the ignore pad attention is true, consider
                # padding. Otherwise attend to all tokens of the unit
                if fill == 0:
                    # add all 0 row
                    curr_row = [0]*total_seq_len
                    decoder_causal_mask.append(curr_row)
                    continue

                # if cross_unit_attention:
                #     # cur_row = copy.deepcopy(
                #     #     first_col) + [0]*(total_seq_len-len(first_col))
                #     cur_row = [0]*total_seq_len if i==0 else copy.deepcopy(decoder_causal_mask[-1])
                # else:
                #     # when ignoring cross attention, all previous units would be zero
                #     # copy the previous row if between a unit, otherwise innitialize with
                #     # all zeros
                curr_row = [
                    0]*total_seq_len if i == 0 else copy.deepcopy(decoder_causal_mask[-1])

                if cross_unit_attention:
                    # update position i of every unit of valid target kp
                    num_units = target_kps_olen # max_kps 
                    for jj in range(num_units):
                        offset_ind = jj*max_kp_len + i
                        curr_row[offset_ind] = fill
                    decoder_causal_mask.append(curr_row)
                else:
                    # update the position
                    offset_ind = kp_i*max_kp_len + i
                    curr_row[offset_ind] = fill
                    decoder_causal_mask.append(curr_row)

        # pad remaining units for decoder
        def LCM(a, b):
            greater = max(a, b)
            smallest = min(a, b)
            for i in range(greater, a*b+1, greater):
                if i % smallest == 0:
                    return i

        pad_units = max_kps-len(target_kps)

        # get the first column of the 2d decoder_causal_mask as the loss weight mask by default
        # loss_weight_mask = [decoder_causal_mask[i][0]
        #                     for i in range(len(decoder_causal_mask))]

        # loss weight mask by default (without phi tokens) excludes the extra padded units
        loss_weight_mask = [1.0]*(target_kps_olen*max_kp_len) + [
            0.0]*((max_kps-target_kps_olen)*max_kp_len)

        # add phi tokens for seq2set so that precision of the ensemble of independant decoding heads dont drop
        # the loss function is downweighted for the pad units or the units with no kps
        # this should give enough feedback to the model to bump up the precision
        if self.add_phi_tokens:
            loss_weight_mask = [1.0]*(max_kps*max_kp_len)

            assert pad_units == 0, f'add phi tokens enabled and pad units>0 (pad units need to be zero in this setting), pad_units={pad_units}, max_kps={max_kps}'
            # prepare dynamic loss weight mask, when actual KPs less than 80% of the padded kps
            # if target_kps_olen < int(0.50*max_kps) and not none_dp:
            #     lcm = LCM(target_kps_olen, max_kps-target_kps_olen)
            #     nontoken_wt_factor = lcm/(max_kps-target_kps_olen)
            #     token_wt_factor = lcm/(target_kps_olen)

            #     loss_weight_mask_factors = [token_wt_factor]*(target_kps_olen*max_kp_len) + [
            #         nontoken_wt_factor]*((max_kps-target_kps_olen)*max_kp_len)
            #     loss_weight_mask = [
            #         val*fac for val, fac in zip(loss_weight_mask, loss_weight_mask_factors)]

            

        # print('loss weight mask: ', loss_weight_mask)
        # print('kps: ', target_kps)

        for _ in range(pad_units*max_kp_len):
            decoder_causal_mask.append([0]*total_seq_len)
        decoder_target_tokens += ['[PAD]']*(pad_units*max_kp_len)
        decoder_input_tokens += ['[PAD]']*(pad_units*max_kp_len)

        # get list of target gold Absent KPs
        target_gold_kps = get_goldakp(target=data_point['tgt'].strip())

        # get input document tokens
        doc_tokens = [data_point['doc']]

        assert (self.truncate_config['len_a']-2) <= self.max_len_encoder, \
            f'''Maximum seq_a length {self.truncate_config['len_a']} 
                cannot be fit within max total len {self.max_len_encoder}'''

        ## Encoder Inputs ##
        # pad to max length
        # add special tokens
        doc_list = ['[CLS]']+doc+['[SEP]']
        encoder_pad_length = self.max_len_encoder - len(doc_list)

        # convert tokens to input_ids for model
        encoder_input_ids = self.model_tokenizer.convert_tokens_to_ids(
            doc_list) + [0]*encoder_pad_length
        encoder_attention_mask = [
            1]*len(doc_list) + [0]*encoder_pad_length

        cross_attention_mask = [
            1]*len(doc_list) + [0]*encoder_pad_length
        
        encoder_mlm_labels_mask = [0]*self.max_len_encoder

        # innitially checking with GT setting
        if self.filter_encoder_hidden_states:
            if self.gt_setting:
                # find positions of the extractive tokens
                labels = self.data_tokenizer.tokenize(
                    data_point['lab'].strip(),
                    truncate=self.truncate_config['len_a'])

            if not self.gt_setting or self.use_pseudo_labels:
                # obtain labels from the extractive model
                doc_pred, doc_prob = self.pkp_inference_pipeline(
                    data_point=data_point, is_train=False)
                # print('doc_pred: ', doc_pred)
                # labels = doc_pred

            if (not self.gt_setting) or (len(set(labels)) == 1 and labels[0] == 'O'):
                # empty line, try to use pseudo labels for extractive tokens
                labels = doc_pred
                # print('\n Using pred labels: ', doc_pred)

            label_list = ['O']+labels+['O']
            # print('label list: ', label_list)
            pred_kps = []  # list of [ex kp word, total tokens, [start, end]]
            b_start = -1
            for token_ind, (pred_label, doc_token) in enumerate(zip(label_list, doc_list)):
                # pairs.append([pred_ind, doc_token])
                if pred_label == 'B':
                    # start recording KP +1 for excluded CLS token
                    pred_kps.append(
                        [doc_token, 1, [token_ind, -1]])
                    b_start = token_ind  # start contiguous check
                elif pred_label in ['X', 'I'] and b_start != -1:
                    # need this to be contiguous (only following B)
                    pred_kps[-1][0] += doc_token.lstrip(
                        "##") if doc_token.startswith("##") else ' '+doc_token
                    # number of words in the keyphrase
                    pred_kps[-1][1] += 1
                elif pred_label == 'O' and b_start != -1:
                    b_start = -1

            for span in pred_kps:
                # end = start + span_length
                span[2][1] = span[2][0]+span[1]
                # print(doc_list[span[2][0]:span[2][1]])

            cross_attention_mask = [0]*self.max_len_encoder
            for pk in pred_kps:
                start, end = pk[-1]
                # print('\ntoken: ', doc_list[start:end])
                for j in range(start, end):
                    cross_attention_mask[j] = 1

            # negated_cross_attention_mask = [
            #     1 if x == 0 else 0 for x in cross_attention_mask]
            
            # prepare inputs and labels for mlm head at encoder side

            
            
            # only mask the inputs in train mode with encoder mlm
            if self.encoder_masking and is_train:
                masked_doc_list = copy.deepcopy(doc_list)
                # modify the input ids to contain masked tokens in locations of extractive tokens
                for pk in pred_kps:
                    start, end = pk[-1]
                    # print('\ntoken: ', start, end)
                    for j in range(start, end):
                        encoder_mlm_labels_mask[j] = 1
                        masked_doc_list[j] = '[MASK]'
                        # also change encoder attention mask at mask positions to 0
                        encoder_attention_mask[j] = 0

                encoder_input_ids = self.model_tokenizer.convert_tokens_to_ids(
                    masked_doc_list) + [0]*(self.max_len_encoder-len(masked_doc_list))
                
            # print('\nencoder_input_ids: ', encoder_input_ids)
            # print('\nencoder attention mask: ', encoder_attention_mask)

        ## Decoder Inputs ##
        decoder_target_ids = self.model_tokenizer.convert_tokens_to_ids(
            decoder_target_tokens)
        decoder_input_ids = self.model_tokenizer.convert_tokens_to_ids(
            decoder_input_tokens)

        # decoder position ids for diversity within independant decoding
        if self.control_codes:
            # seq2set paper setting
            decoder_position_ids = [x for x in range(0, max_kp_len)]*max_kps            
            decoder_token_type_ids = [[i for _ in range(max_kp_len)] for i in range(max_kps)]
            decoder_token_type_ids = [x for item in decoder_token_type_ids for x in item]
        else:
            # our setting
            decoder_position_ids = list(range(0,len(decoder_input_ids)))
            decoder_token_type_ids = [0 for _ in range(0,len(decoder_input_ids))]

        # print(len(decoder_target_ids))
        # print(len(decoder_input_ids))

        # print('\nencoder_input_ids: ', encoder_input_ids)
        # print('\nencoder mlm labels mask: ',encoder_mlm_labels_mask)

        # print('\n: Decoder Causal Mask: \n')
        # for x in decoder_causal_mask:
        #     print(x)

        # print('\n input ids: ', decoder_input_ids)
        # print('\n')
        ds = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'encoder_mlm_labels_mask': encoder_mlm_labels_mask,
            'cross_attention_mask': cross_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_causal_mask,
            'decoder_position_ids': decoder_position_ids,
            'decoder_token_type_ids': decoder_token_type_ids,
            'labels': decoder_target_ids,
            'doc': doc_tokens,
            'gold_akp': target_gold_kps,
            'loss_weight_mask': loss_weight_mask
        }

        return ds


# proposed emb2set pipeline
class AKPEmb2SetPipeline:
    def __init__(self,
                 data_tokenizer,
                 kp_tokenizer,
                 model_tokenizer,
                 truncate_config,
                 decoder_output_config,
                 max_len,
                 inv_label_map,
                 label_map,
                 model_load_config,
                 vanilla_nar,
                 cross_unit_attention,
                 mean_reduce_repeat,
                 **kwargs):
        self.data_tokenizer = data_tokenizer
        self.kp_tokenizer = kp_tokenizer
        self.model_tokenizer = model_tokenizer
        self.truncate_config = truncate_config
        self.max_len_encoder = max_len
        self.decoder_output_config = decoder_output_config
        self.inv_label_map = inv_label_map
        self.label_map = label_map
        self.model_load_config = model_load_config
        self.vanilla_nar = vanilla_nar
        self.cross_unit_attention = cross_unit_attention
        self.mean_reduce_repeat = mean_reduce_repeat

        # innitialize encoder pipeline
        self.encoder_pipeline = PKPBasePipeline(
            data_tokenizer=self.data_tokenizer,
            model_tokenizer=self.model_tokenizer,
            truncate_config=self.truncate_config,
            max_len=self.max_len_encoder,
            label_map=None
        )

        # innitialize decoder pipeline
        self.decoder_pipeline = AKPSeq2SetPipeline(
            data_tokenizer=self.data_tokenizer,
            model_tokenizer=self.model_tokenizer,
            kp_tokenizer=self.kp_tokenizer,
            truncate_config=self.truncate_config,
            decoder_output_config=self.decoder_output_config,
            max_len=max_len,
            cross_unit_attention=False,
            ignore_pad_attention=False,
            add_phi_tokens=False,
            inv_label_map=self.inv_label_map,
            label_map=self.label_map,
            filter_encoder_hidden_states=None,
            gt_setting=None,
            use_pseudo_labels=None,
            model_load_config=None
        )
        self.enc_device = torch.device('cuda')
        self.encoder_model = self._load_encoder(config=self.model_load_config)
        self.encoder_model.to(self.enc_device)

    def _load_encoder(self, config):
        model = pkp_base_loader(config=config)
        return model

    def _compute_2d_mask_from_1d(self, oned_mask, max_units, max_unit_length):
        max_kps, max_seq_len = max_units, max_unit_length
        decoder_token_attention_mask_2d = []
        for i in range(max_kps):
            slice = oned_mask[i*max_seq_len:(i+1)*max_seq_len]
            # find last non zero index
            one_pos = [pos for pos, elem in enumerate(slice) if elem != 0]
            last_one_pos = one_pos[-1] if one_pos else -1
            if last_one_pos != -1:
                fill = 1
            else:
                fill = 0
                last_one_pos = max_seq_len-1
            for j in range(max_seq_len):
                row = [0]*max_seq_len*max_kps
                if j <= last_one_pos:
                    row = [0]*i*max_seq_len + [fill]*(last_one_pos+1) + [0]*(
                        max_seq_len*(max_kps-(i+1)) + max_seq_len-(last_one_pos+1))
                decoder_token_attention_mask_2d.append(row)
        return decoder_token_attention_mask_2d

    def __call__(self, data_point, is_train):
        # encoder input
        pre_inp = self.encoder_pipeline(data_point=data_point)
        self.encoder_model.eval()
        max_seq_len = self.decoder_output_config['max_kp_len']
        max_kps = self.decoder_output_config['max_kps']

        # start = time.time()
        with torch.no_grad():
            enc_inp_ids = torch.tensor(
                pre_inp['input_ids']).unsqueeze(0).to(self.enc_device)
            enc_tok_ids = torch.tensor(
                pre_inp['token_type_ids']).unsqueeze(0).to(self.enc_device)
            enc_attn_mask = torch.tensor(
                pre_inp['attention_mask']).unsqueeze(0).to(self.enc_device)
            doc_text = pre_inp['doc'][0]

            model_out = self.encoder_model(
                input_ids=enc_inp_ids,
                token_type_ids=enc_tok_ids,
                attention_mask=enc_attn_mask,
                labels=None,
                output_hidden_states=True,
                return_dict=True
            )
            # get last hidden state
            all_hidden_states = model_out['hidden_states']
            hidden_state = all_hidden_states[-1]
            logits = model_out['logits']
            # ignoring [CLS] token
            doc_prob = (logits.detach().cpu().numpy())[0][1:]
            doc_pred = ((torch.argmax(logits, dim=2)
                         ).detach().cpu().numpy())[0][1:]

        # end = time.time()
        # print('model forward time: ', end-start)

            # start = time.time()
            pred_kps = []
            doc_text_toks = doc_text.strip().split(" ")
            b_start = -1
            for token_ind, (pred_ind, doc_token) in enumerate(zip(doc_pred, doc_text_toks)):
                # pairs.append([pred_ind, doc_token])
                pred_label = self.inv_label_map[pred_ind]
                token_score = doc_prob[token_ind][pred_ind]
                if pred_label == 'B':
                    # start recording KP +1 for excluded CLS token
                    pred_kps.append(
                        [doc_token, token_score, 1, [token_ind+1, -1]])
                    b_start = token_ind  # start contiguous check
                elif pred_label in ['X', 'I'] and b_start != -1:
                    # need this to be contiguous (only following B)
                    pred_kps[-1][0] += doc_token.lstrip(
                        "##") if doc_token.startswith("##") else ' '+doc_token
                    pred_kps[-1][1] += token_score
                    # number of words in the keyphrase
                    pred_kps[-1][2] += 1
                elif pred_label == 'O' and b_start != -1:
                    b_start = -1

            # create span embeddings
            for span in pred_kps:
                # end = start + span_length
                span[3][1] = span[3][0]+span[2]
                # get the span embeddings
                span_emb = hidden_state[0, span[3]
                                        [0]:span[3][1], :].detach().cpu()

                if self.mean_reduce_repeat:
                    span_emb = torch.mean(span_emb, dim=0).unsqueeze(
                        0).repeat(max_seq_len, 1)
                    # print(span_emb.shape)

                span.append(span_emb)

                # get corresponding tokens
                tokens = [doc_text_toks[span[3][0]-1+i]
                          for i in range(0, span[2])]
                # print('\ntokens: ', tokens)
                span.append(tokens)

        # print('pred kps: ',pred_kps)

        pred_kps = sorted(pred_kps, key=lambda x: -(x[1]/float(x[2])))

        # truncate for max unit lengths and max kps
        pred_kps = pred_kps[:max_kps]
        for span in pred_kps:
            span[4] = span[4][:max_seq_len]
            # truncate unit to max_seq_len
            span[2] = min(max_seq_len, span[2])
            if self.mean_reduce_repeat:
                # as mean embedding is repeated max_seq_len times
                span[2] = max_seq_len

            # add padding
            pad_length = max_seq_len-len(span[4])
            pad_tensor = torch.zeros(pad_length, 768)
            span[4] = torch.cat([span[4], pad_tensor], dim=0)

            # truncate and pad ex kp tokens
            span[5] = ['[CLS]']+span[5][:max_seq_len-2]+['[SEP]']
            span[5] = span[5] + ['[PAD]']*(max_seq_len-len(span[5]))

        # pad to maximum units
        pad_unit_length = max_kps-len(pred_kps)
        for _ in range(pad_unit_length):
            pad_unit = ['', 0.0, 0, [-1, -1],
                        torch.zeros(max_seq_len, 768), ['[PAD]']*max_seq_len]
            pred_kps.append(pad_unit)

        all_unit_embs = []
        for span in pred_kps:
            all_unit_embs.append(span[4])

        # prepare attention mask for input_embeds
        decoder_attention_mask = []
        for span in pred_kps:
            kp_len = span[2]
            unit_mask = [1]*kp_len + [0]*(max_seq_len-kp_len)
            decoder_attention_mask.extend(unit_mask)
        # print('unit attention mask: ', decoder_attention_mask)

        decoder_input_embeds = torch.stack(all_unit_embs, dim=0)

        # print('pred kps: ',pred_kps)
        # prepare decoder input ids

        # prepare attention mask for extractive tokens (instead of embeds input)
        decoder_token_attention_mask = []
        decoder_input_ids = []
        for span in pred_kps:
            # print('\nspan tokens: ', span[5])
            span[5] = self.model_tokenizer.convert_tokens_to_ids(span[5])
            # print('\nspan tokens ids: ', span[5])

            attn_len = 0 if span[2] == 0 else min(span[2]+2, max_seq_len)
            unit_mask = [1]*attn_len + [0]*(max_seq_len-attn_len)
            decoder_token_attention_mask.extend(unit_mask)
            decoder_input_ids.extend(span[5])

        # prepare 2d unit mask (non causal)
        # use 1d mask when no cross unit attention applied
        if not self.cross_unit_attention:
            decoder_token_attention_mask = self._compute_2d_mask_from_1d(oned_mask=decoder_token_attention_mask,
                                                                         max_units=max_kps,
                                                                         max_unit_length=max_seq_len)
            decoder_attention_mask = self._compute_2d_mask_from_1d(oned_mask=decoder_attention_mask,
                                                                   max_units=max_kps,
                                                                   max_unit_length=max_seq_len)

        # print('\ndecoder 1d token attention mask: ')
        # print(decoder_token_attention_mask)
        # print('\n2d token attention mask: ')
        # for x in decoder_token_attention_mask_2d:
        #     print(x)

        # overriding input ids when cross attentino layers are passed
        if self.vanilla_nar:
            # if vanilla nar, the input ids are all mask tokens
            # the last hidden state of encoder is passed to cross-
            # attention layers of the decoder
            # isDecoder flag is set to True
            decoder_token_attention_mask = [1]*(max_seq_len*max_kps)
            decoder_input_ids = self.model_tokenizer.convert_tokens_to_ids(
                ['[MASK]']*(max_seq_len*max_kps))

        # get list of target gold Absent KPs
        target_gold_kps = get_goldakp(target=data_point['tgt'].strip())

        # decoder labels
        dec_label_data = self.decoder_pipeline(data_point=data_point)
        decoder_target_ids = dec_label_data['labels']

        # print('decoder input embeds: ', torch.tensor(decoder_input_embeds).shape)
        # print('decoder attention mask: ',torch.tensor(decoder_attention_mask).shape)
        # print('labels: ',torch.tensor(decoder_target_ids).shape)
        # print('labels: ', decoder_target_ids)
        # print('pred akp: ',pred_kps)
        # print('decoder input ids: ', decoder_input_ids)
        # print('decoder token attention mask: ',decoder_token_attention_mask)

        ds = {
            'decoder_input_ids': decoder_input_ids,
            'decoder_token_attention_mask': decoder_token_attention_mask,
            'decoder_input_embeds': decoder_input_embeds,
            'decoder_attention_mask': decoder_attention_mask,
            'encoder_hidden_states': hidden_state[0],
            'encoder_attention_mask': enc_attn_mask,
            'labels': decoder_target_ids,
            'doc': pre_inp['doc'],
            'gold_akp': target_gold_kps
        }
        # print('\ninput tokens: ',self.model_tokenizer.batch_decode(np.array(decoder_input_ids).reshape(max_kps, max_seq_len)))
        # print('\nlabels: ',self.model_tokenizer.batch_decode(np.array(decoder_target_ids).reshape(max_kps, max_seq_len)))

        # end = time.time()
        # print('full dataset fetch item time: ', end-start)

        return ds


class ExpEmb2SetPipeline:
    def __init__(self,
                 **kwargs):
        from transformers import BertModel, BertTokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda')
        self.model.to(self.device)

    def __call__(self, data_point, is_train):
        doc = data_point['doc']
        target = data_point['tgt']
        label = data_point['lab']

        doc_list = ['[CLS]']+doc.strip().split(" ")+['[SEP]']
        label_list = ['O']+label.strip().split(" ")+['O']

        # convert tokens to ids
        ids = self.tokenizer.convert_tokens_to_ids(doc_list)
        assert len(ids) == len(
            label_list), '# doc tokens not equal to # of labels'

        # key: Ex KP , value: [start_ind, end_ind] of tokens of all words in that key phrase
        ex_kps = {}
        # get the KP spans
        ex_kp, start = '', -1

        for ind, doc_label in enumerate(label_list):
            word = doc_list[ind]
            if doc_label == 'B':
                start = ind
                ex_kp += word
            elif doc_label in ['X', 'I']:  # sub-word or inside kp
                if word.startswith('##'):
                    ex_kp += word.lstrip('##')
                else:  # new word, append with space
                    ex_kp += ' '+word
            elif doc_label == 'O' and start != -1:
                # record the token span
                ex_kps[ex_kp] = ex_kps.setdefault(ex_kp, [])+[[start, ind]]
                ex_kp, start = '', -1  # reset ex_kp and start

        # if last kp extends till end
        if doc_label != 'O':
            ex_kps[ex_kp] = ex_kps.setdefault(ex_kp, [])+[[start, ind]]

        ### Absent key phrase processing ####
        # add noise at end of line
        doc_line_list = ['[CLS]']+doc.strip().split(" ")+['[SEP]']
        ab_line_list = target.strip().split(" ") + ['[SEP]']

        offset = len(doc_line_list)
        ab_kp = {}

        start, ab_word, ab_token_list = 0, '', []
        for ind, tok in enumerate(ab_line_list):
            if tok == ';':
                ab_kp[ab_word] = ab_kp.setdefault(
                    ab_word, [])+[[start+offset, ind+offset, ab_token_list]]
                start, ab_word, ab_token_list = ind+1, '', []
            elif tok != '[SEP]':
                if tok.startswith('##'):
                    ab_word += tok.lstrip('##')
                else:
                    ab_word += ' '+tok
                ab_token_list.append(tok)

        # add final kp
        ab_kp[ab_word] = ab_kp.setdefault(
            ab_word, [])+[[start+offset, ind+offset, ab_token_list]]
        doc_line_list = doc_line_list + ab_line_list

        print('ex_kps: ', ex_kps)
        print('ab_kps: ', ab_kp)

        self.model.eval()
        with torch.no_grad():
            tokens = doc_line_list
            # input_ids
            tok_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(tokens)).to(self.device)
            # token_type_ids
            token_type_ids = torch.tensor(
                [0 for _ in range(len(tokens))]).to(self.device)
            # attention_mask
            attention_mask = torch.tensor(
                [1 for _ in range(len(tokens))]).to(self.device)
            inp_dict = {
                'input_ids': tok_ids.unsqueeze(dim=0),  # add batch dim
                'token_type_ids': token_type_ids.unsqueeze(dim=0),
                'attention_mask': attention_mask.unsqueeze(dim=0)
            }
            model_out = self.model(**inp_dict)
            # last hidden state
            embs = model_out['last_hidden_state']
            # print('model_out shape: ',embs.shape)

            # break
            ex_kp = ex_kps
            # ab_kp =
            # all extractive and abstractive key-phrase embeddings including duplicate positions
            ekp_embs, akp_embs = [], []

            ex_tokens, ab_tokens = [], []
            for kp_e in ex_kp:
                for (s1, e1) in ex_kp[kp_e]:
                    ex_tokens.append(kp_e)
                    extracted_emb = embs[:, s1:e1, :]
                    # print(extracted_emb.shape, start, end, len(lines[line_ind].split(" "))+2)
                    mean_emb_e = torch.mean(extracted_emb, dim=1)
                    ekp_embs.append(mean_emb_e)

            target_toks = []
            for kp_a in ab_kp:
                for (s2, e2, toks) in ab_kp[kp_a]:
                    ab_tokens.append(kp_a)

                    mod_toks = '[CLS]'+toks[:3]+'[SEP]'
                    mod_toks += '[PAD]'*(5-len(mod_toks))
                    target_toks.extend(mod_toks)

                    abstracted_emb = embs[:, s2:e2, :]
                    mean_emb_a = torch.mean(abstracted_emb, dim=1)
                    akp_embs.append(mean_emb_a)

            # pad target for remaining units
            pad_units = (8*5-len(target_toks))*['[PAD]']
            target_toks += pad_units
            target_tok_ids = self.tokenizer.convert_tokens_to_ids(target_toks)

            # print(ex_tokens, ab_tokens)
            # ex_ab_tokenpairs = itertools.product(ex_tokens, ab_tokens)
            ekp_embs = torch.stack(ekp_embs, dim=1).squeeze(0).detach().cpu()
            akp_embs = torch.stack(akp_embs, dim=1).squeeze(0).detach().cpu()

            ekp_embs = ekp_embs[:8]
            akp_embs = akp_embs[:8]

            orig_e = len(ekp_embs)
            pad_length_e = 8-len(ekp_embs)
            pad_tensor_e = torch.zeros(pad_length_e, 768)

            pad_length_a = 8-len(akp_embs)
            pad_tensor_a = torch.zeros(pad_length_a, 768)

            ekp_embs = torch.cat([ekp_embs, pad_tensor_e], dim=0)
            akp_embs = torch.cat([akp_embs, pad_tensor_a], dim=0)
            attention_mask = [1]*orig_e + [0]*pad_length_e

            target_gold_kps = get_goldakp(target=data_point['tgt'].strip())

            ds = {
                'ekp_embs': ekp_embs,
                'akp_embs': akp_embs,
                'labels': target_tok_ids,
                'attention_mask': attention_mask,
                'doc': [data_point['doc']],
                'gold_kps': target_gold_kps,
            }
            # print(ekp_embs.shape)
            # print(akp_embs.shape)
            # print(attention_mask.shape)
            return ds


def main():
    pipeline = ExpEmb2SetPipeline()

    from dataset import KPEDataset
    project_path = '/home/thomased/work/codebase'
    doc_path = f'{project_path}/coopsummer2023/UniKP/UniKeyphrase/processed/kp20k.test.seq.in'
    doc_label_path = f'{project_path}/coopsummer2023/UniKP/UniKeyphrase/processed/kp20k.test.seq.out'
    doc_abs_path = f'{project_path}/coopsummer2023/UniKP/UniKeyphrase/processed/kp20k.test.absent'

    # test the dataset
    dataset = KPEDataset(
        document_file=doc_path,
        label_file=doc_label_path,
        target_file=doc_abs_path,
        preprocess_pipeline=pipeline,
        skip_none=True)

    check_ind = 1
    x = dataset.__getitem__(check_ind)
    # print(x)


if __name__ == '__main__':
    main()

# from transformers import BertForTokenClassification
from torch.utils.data import DataLoader
from dataset.dataset import KPEDataset
from tokenization.tokenization import WhitespaceTokenizer
from transformers import BertTokenizer
import torch
from models.model_maps import maps
import os
import argparse
import json
import pandas as pd
from torch.nn import DataParallel
import copy


def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    # get the configs
    '''
    example usage:
    python ./test.py --config './configs/test.json'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = config_parser(args.config)

    # create label maps
    label_map, inv_label_map = {}, {}
    label_list = ['O', 'B', 'I', 'X']
    for ind, lab in enumerate(label_list):
        label_map[lab] = ind
        inv_label_map[ind] = lab

    # get model specific collators and pipelines
    model_loader = maps[config['model_map_key']]['model_loader']
    collator = maps[config['model_map_key']]['collator']
    tester = maps[config['model_map_key']]['validator']
    data_pipeline = maps[config['model_map_key']]['data_pipeline']
    data_tokenizer = maps[config['model_map_key']]['data_tokenizer']
    kp_tokenizer = maps[config['model_map_key']]['kp_tokenizer']

    # innitialize tokenizers
    data_tokenizer = data_tokenizer() if data_tokenizer else None
    kp_tokenizer = kp_tokenizer() if kp_tokenizer else None
    model_tokenizer = BertTokenizer.from_pretrained(config['tokenizer']['path_or_name'],
                                                    do_lower_case=True)

    # Load Checkpoint
    config['model']['tokenizer_vocab_size'] = len(model_tokenizer.vocab)
    model = model_loader(config=config, tokenizer=model_tokenizer)

    device = torch.device(config['run_params']['device'])
    model.to(device)
    model = DataParallel(model)

    result_dict = {'dataset': [],
                   'P@M': [],
                   'R@M': [],
                   'F1@M': [],
                   'P@K': [],
                   'R@K': [],
                   'F1@K': [],
                   'P@O': [],
                   'R@O': [],
                   'F1@O': []}

    result_dict_macro = copy.deepcopy(result_dict)

    significance_scores_dict_macro = {'dataset': [],
                                      'F1@M': [],
                                      'F1@K': [],
                                      'F1@O': []}

    if 'seq2set' in config['model_map_key']:
        max_kps = config['model']['max_kps']
        result_dict_heads = [copy.deepcopy(
            result_dict) for _ in range(max_kps)]
        result_dict_macro_heads = [copy.deepcopy(
            result_dict) for _ in range(max_kps)]

    # instantiate pipeline
    pipeline = data_pipeline(
        data_tokenizer=data_tokenizer,
        model_tokenizer=model_tokenizer,
        kp_tokenizer=kp_tokenizer,
        truncate_config=config['tokenizer']['trunc_conf'],
        label_map=label_map,
        inv_label_map=inv_label_map,
        max_len=config['tokenizer']['max_len'],
        max_len_d=config['tokenizer']['max_len_d'],
        cross_unit_attention=config['run_params']['cross_unit_attention'],
        decoder_output_config={
            'max_kp_len': config['model']['max_kp_len'], 'max_kps': config['model']['max_kps']},
        ignore_pad_attention=config['run_params']['ignore_pad_attention'],
        add_phi_tokens=config['run_params']['add_phi_tokens'],
        model_load_config={
            'project_path': config['project_path'], 'model': config['model']},
        vanilla_nar=config['run_params']['vanilla_nar'],
        mean_reduce_repeat=config['run_params']['mean_reduce_repeat'],
        filter_encoder_hidden_states=config['run_params']['filter_extkp_hidden_states'],
        gt_setting=config['run_params']['gt_setting'],
        use_pseudo_labels=config['run_params']['use_pseudo_labels'],
        encoder_masking=config['model']['encoder_mlm'],
        control_codes=True if int(config['model']['decoder_type_vocab_size'])>2 else False
    )
    # save_dir = os.path.join(
    #     config['project_path'], os.path.dirname(config['output_path']))
    save_dir = os.path.join(
        config['project_path'], config['output_path'])

    for ds in config['datasets']:
        result_dict['dataset'].append(ds['name'])
        result_dict_macro['dataset'].append(ds['name'])
        significance_scores_dict_macro['dataset'].append(ds['name'])

        if 'seq2set' in config['model_map_key']:
            for kp_ind in range(max_kps):
                result_dict_heads[kp_ind]['dataset'].append(ds['name'])
                result_dict_macro_heads[kp_ind]['dataset'].append(ds['name'])

        test_dataset = KPEDataset(
            document_file=os.path.join(config['project_path'], ds['doc']),
            label_file=os.path.join(config['project_path'], ds['label']),
            target_file=os.path.join(config['project_path'], ds['target']),
            preprocess_pipeline=pipeline,
            skip_none=config['run_params']['skip_none'],
            is_train=False
        )

        # test data loader
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config['run_params']['batch_size'],
            collate_fn=collator,
            pin_memory=True
        )

        ##### EVALUATION #####
        res = tester(
            val_loader=test_loader,
            model=model,
            device=device,
            inv_label_map=inv_label_map,
            top_k=config['run_params']['top_k'],
            tokenizer=model_tokenizer,
            thresh=config['run_params']['thresh'],
            run_conf={
                'max_kp_len': config['model']['max_kp_len'], 'max_kps': config['model']['max_kps']},
            save_dir=save_dir+'_'+ds['name']
        )

        if 'seq2set' in config['model_map_key']:
            res, res_units, significance_scores = res
            num_res_units = len(res_units[0])
            assert num_res_units == len(res_units[1])
        else:  # for pkp and akp seq2seq
            res, significance_scores = res

        # micro results
        result_dict['P@M'].append(round(res[0]['p']*100, 2))
        result_dict['R@M'].append(round(res[0]['r']*100, 2))
        result_dict['F1@M'].append(round(res[0]['f']*100, 2))
        result_dict['P@K'].append(round(res[0]['p_k']*100, 2))
        result_dict['R@K'].append(round(res[0]['r_k']*100, 2))
        result_dict['F1@K'].append(round(res[0]['f_k']*100, 2))
        result_dict['P@O'].append(round(res[0]['p_o']*100, 2))
        result_dict['R@O'].append(round(res[0]['r_o']*100, 2))
        result_dict['F1@O'].append(round(res[0]['f_o']*100, 2))

        # macro_results
        result_dict_macro['P@M'].append(round(res[1]['p']*100, 2))
        result_dict_macro['R@M'].append(round(res[1]['r']*100, 2))
        result_dict_macro['F1@M'].append(round(res[1]['f']*100, 2))
        result_dict_macro['P@K'].append(round(res[1]['p_k']*100, 2))
        result_dict_macro['R@K'].append(round(res[1]['r_k']*100, 2))
        result_dict_macro['F1@K'].append(round(res[1]['f_k']*100, 2))
        result_dict_macro['P@O'].append(round(res[1]['p_o']*100, 2))
        result_dict_macro['R@O'].append(round(res[1]['r_o']*100, 2))
        result_dict_macro['F1@O'].append(round(res[1]['f_o']*100, 2))

        # significance scores macro
        significance_scores_dict_macro['F1@M'].append(significance_scores[0])
        significance_scores_dict_macro['F1@K'].append(significance_scores[1])
        significance_scores_dict_macro['F1@O'].append(significance_scores[2])

        # update all heads
        if 'seq2set' in config['model_map_key']:
            # iterate through all units and populate scores
            for res_u_ind in range(num_res_units):
                res_u0 = res_units[0][res_u_ind]
                result_dict_heads[res_u_ind]['P@M'].append(
                    round(res_u0['p']*100, 2))
                result_dict_heads[res_u_ind]['R@M'].append(
                    round(res_u0['r']*100, 2))
                result_dict_heads[res_u_ind]['F1@M'].append(
                    round(res_u0['f']*100, 2))
                result_dict_heads[res_u_ind]['P@K'].append(
                    round(res_u0['p_k']*100, 2))
                result_dict_heads[res_u_ind]['R@K'].append(
                    round(res_u0['r_k']*100, 2))
                result_dict_heads[res_u_ind]['F1@K'].append(
                    round(res_u0['f_k']*100, 2))
                result_dict_heads[res_u_ind]['P@O'].append(
                    round(res_u0['p_o']*100, 2))
                result_dict_heads[res_u_ind]['R@O'].append(
                    round(res_u0['r_o']*100, 2))
                result_dict_heads[res_u_ind]['F1@O'].append(
                    round(res_u0['f_o']*100, 2))

                # macro_results
                res_u1 = res_units[1][res_u_ind]
                result_dict_macro_heads[res_u_ind]['P@M'].append(
                    round(res_u1['p']*100, 2))
                result_dict_macro_heads[res_u_ind]['R@M'].append(
                    round(res_u1['r']*100, 2))
                result_dict_macro_heads[res_u_ind]['F1@M'].append(
                    round(res_u1['f']*100, 2))
                result_dict_macro_heads[res_u_ind]['P@K'].append(
                    round(res_u1['p_k']*100, 2))
                result_dict_macro_heads[res_u_ind]['R@K'].append(
                    round(res_u1['r_k']*100, 2))
                result_dict_macro_heads[res_u_ind]['F1@K'].append(
                    round(res_u1['f_k']*100, 2))
                result_dict_macro_heads[res_u_ind]['P@O'].append(
                    round(res_u1['p_o']*100, 2))
                result_dict_macro_heads[res_u_ind]['R@O'].append(
                    round(res_u1['r_o']*100, 2))
                result_dict_macro_heads[res_u_ind]['F1@O'].append(
                    round(res_u1['f_o']*100, 2))

    df = pd.DataFrame(data=result_dict, columns=list(result_dict.keys()))
    df.to_csv(os.path.join(config['project_path'],
                           config['output_path']+'_micro'), index=False)

    df = pd.DataFrame(data=result_dict_macro,
                      columns=list(result_dict_macro.keys()))
    df.to_csv(os.path.join(config['project_path'],
                           config['output_path']+'_macro'), index=False)

    # write significance scores
    df = pd.DataFrame(data=significance_scores_dict_macro,
                      columns=list(significance_scores_dict_macro.keys()))
    df.to_csv(os.path.join(config['project_path'],
                           config['output_path']+'_significance'), index=False)

    # add results for all unit heads separately for seq2set setting
    # update all heads
    if 'seq2set' in config['model_map_key']:
        # iterate through all units and populate scores
        for res_u_ind in range(num_res_units):
            # print(f'\ndata {res_u_ind}: ', result_dict_heads[res_u_ind])
            df = pd.DataFrame(data=result_dict_heads[res_u_ind], columns=list(
                result_dict_heads[res_u_ind].keys()))
            df.to_csv(os.path.join(config['project_path'],
                                   config['output_path']+f'_micro_unit{res_u_ind}'), index=False)

            df = pd.DataFrame(data=result_dict_macro_heads[res_u_ind],
                              columns=list(result_dict_macro_heads[res_u_ind].keys()))
            df.to_csv(os.path.join(config['project_path'],
                                   config['output_path']+f'_macro_unit{res_u_ind}'), index=False)


if __name__ == '__main__':
    main()

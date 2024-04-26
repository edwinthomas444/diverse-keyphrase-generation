# from transformers import BertForTokenClassification
from torch.utils.data import DataLoader
from dataset.dataset import KPEDataset
from tokenization.tokenization import WhitespaceTokenizer
from transformers import BertTokenizer
import torch
from models.model_maps import maps
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import torch.distributed as dist
import os
import argparse
import json


def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# custom collate function


def main():
    # get the configs
    '''
    example usage:
    python ./train.py --config './configs/train.json'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ddp', action='store_true',
                        help='If set runs with DDP, defaults to DP')
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    config = config_parser(args.config)

    if args.ddp:
        n_gpu = torch.cuda.device_count()
        print(f'Innitializing DDP: num gpus {n_gpu}')
        dist.init_process_group(backend='nccl')
        gpu_rank = torch.distributed.get_rank()
        args.local_rank = gpu_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device(config['run_params']['device'])

    # create label maps
    label_map, inv_label_map = {}, {}
    label_list = ['O', 'B', 'I', 'X']
    for ind, lab in enumerate(label_list):
        label_map[lab] = ind
        inv_label_map[ind] = lab

    if args.ddp and args.local_rank != 0:
        # block all non-base processes
        dist.barrier()

    # get model specific collators and pipelines
    model_loader = maps[config['model_map_key']]['model_loader']
    collator = maps[config['model_map_key']]['collator']
    trainer = maps[config['model_map_key']]['trainer']
    data_pipeline = maps[config['model_map_key']]['data_pipeline']
    data_tokenizer = maps[config['model_map_key']]['data_tokenizer']
    kp_tokenizer = maps[config['model_map_key']]['kp_tokenizer']

    # innitialize tokenizers
    data_tokenizer = data_tokenizer() if data_tokenizer else None
    kp_tokenizer = kp_tokenizer() if kp_tokenizer else None
    model_tokenizer = BertTokenizer.from_pretrained(config['tokenizer']['path_or_name'],
                                                    do_lower_case=True)

    # instantiate model (from hugging face checkpoint (or state_dict=null) OR from local checkpoint (state_dict=path of .bin))
    config['model']['tokenizer_vocab_size'] = len(model_tokenizer.vocab)
    model = model_loader(config=config, tokenizer=model_tokenizer)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n Total parameters: ', pytorch_total_params)
    # exit(0)


    if args.ddp and args.local_rank == 0:
        # at this point, even the base process is blocked
        # the other processes can use cached data and all
        # n_gpu processes are unblocked after this point
        dist.barrier()

    model.to(device)
    if args.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        print('Innitializing DDP Model')
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    else:
        print('Innitializing DP Model')
        # using DataParallel
        model = DataParallel(model)

    # instantiate pipeline (all pipelines should follow same signature)
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
        vanilla_nar = config['run_params']['vanilla_nar'],
        mean_reduce_repeat = config['run_params']['mean_reduce_repeat'],
        filter_encoder_hidden_states = config['run_params']['filter_extkp_hidden_states'],
        gt_setting = config['run_params']['gt_setting'],
        use_pseudo_labels = config['run_params']['use_pseudo_labels'],
        encoder_masking = config['model']['encoder_mlm'],
        control_codes=True if int(config['model']['decoder_type_vocab_size'])>2 else False
    )

    # instantiate Datasets
    split = 'train'
    train_dataset = KPEDataset(
        document_file=os.path.join(
            config['project_path'], config['dataset'][split]['doc']),
        label_file=os.path.join(
            config['project_path'], config['dataset'][split]['label']),
        target_file=os.path.join(
            config['project_path'], config['dataset'][split]['target']),
        preprocess_pipeline=pipeline,
        skip_none=config['run_params']['skip_none'],
        is_train=True
    )

    split = 'validation'
    val_dataset = KPEDataset(
        document_file=os.path.join(
            config['project_path'], config['dataset'][split]['doc']),
        label_file=os.path.join(
            config['project_path'], config['dataset'][split]['label']),
        target_file=os.path.join(
            config['project_path'], config['dataset'][split]['target']),
        preprocess_pipeline=pipeline,
        skip_none=config['run_params']['skip_none'],
        is_train=False
    )

    # train data loader
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = None
    else:
        train_sampler = None  # RandomSampler(train_dataset, replacement=False)
        shuffle = True

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['run_params']['batch_size']//n_gpu if args.ddp else config['run_params']['batch_size'],
        sampler=train_sampler,
        shuffle=shuffle,  # enable shuffle of data
        collate_fn=collator,
        pin_memory=True
    )

    # val data loader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['run_params']['batch_size_val'],
        collate_fn=collator,
        pin_memory=True
    )

    # Innitialize optimizer
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    # get total steps (sum of steps in all epochs)
    epochs = config['run_params']['epochs']
    gradient_accumulation_steps = config['optimizer']['gradient_accumulation_steps']
    warmup_proportion = config['optimizer']['warmup_proportion']

    total_steps = int(len(train_loader)*epochs/gradient_accumulation_steps)
    num_warmup_steps = int(total_steps*warmup_proportion)
    print(
        f'\nTotal Steps: {total_steps} | Total Warmup Steps: {num_warmup_steps}')

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config['optimizer']['learning_rate'],
        correct_bias=False
    )
    # linear warmupscheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    # set display frequency
    display_frequency = config['run_params']['display_freq']

    # checkpoint save path
    save_dir = os.path.join(
        config['project_path'], config['output_path'])

    val_conf = {
        'top_k': config['run_params']['top_k'],
        'thresh': config['run_params']['thresh'],
        'batch_size_val': config['run_params']['batch_size_val']
    }


    # invoke trainer
    trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=model_tokenizer,
        device=device,
        inv_label_map=inv_label_map,
        display_freq=display_frequency,
        val_conf=val_conf,
        save_dir=save_dir,
        run_conf={
            'max_kp_len': config['model']['max_kp_len'], 'max_kps': config['model']['max_kps']},
        # Namespace object to dict, args for passing the local rank
        args=vars(args)
    )


if __name__ == '__main__':
    main()

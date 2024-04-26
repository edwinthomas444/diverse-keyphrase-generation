import torch
from tqdm import trange, tqdm
from evaluation.present_kp_eval import evaluate_present, display_result_present
from evaluation.absent_kp_eval import evaluate_absent, display_result_absent, evaluate_absent_all_heads
import os

from transformers import LogitsProcessor, LogitsProcessorList

# Custom logits processor to implement seq2set based decoding


class UnitLogitsProcessor(LogitsProcessor):

    def __init__(self, max_kp_len, cls_token_id):
        self.max_kp_len = max_kp_len
        self.cls_token_id = cls_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len % self.max_kp_len == 0:
            for i in self.cls_token_id:
                scores[:, i] = 10000  # +float("inf")
        return scores


def save_checkpoint(
        save_dir,
        model,
        epoch
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ep_offset = model.module.config.epoch_offset if hasattr(
        model, 'module') else model.config.epoch_offset
    epoch = epoch+ep_offset
    print('\n saving model epoch number: ', epoch)

    output_model_file = os.path.join(save_dir, f"model.{epoch}.bin")
    output_config_file = os.path.join(save_dir, f"config.json")

    model.eval()
    model_save = model.module if hasattr(model, 'module') else model
    config_save = model.module.config if hasattr(
        model, 'module') else model.config
    # save model checkpoint
    torch.save(model_save.state_dict(), output_model_file)
    # save config
    config_save.to_json_file(output_config_file)


def pke_base_trainer(
        model,
        train_loader,
        val_loader,
        epochs,
        device,
        optimizer,
        scheduler,
        inv_label_map,
        display_freq,
        val_conf,
        save_dir,
        **kwargs
):
    for ep in trange(0,
                     epochs,
                     desc="Epoch",
                     disable=kwargs['args']['local_rank'] != 0):
        #### TRAIN #####
        train_bar = tqdm(
            train_loader,
            desc='Iter (loss=X.XXX)',
            disable=kwargs['args']['local_rank'] != 0
        )
        model.train()
        for train_step, batch in enumerate(train_bar):
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:4], batch[4:]
            inp = [tens.to(device) for tens in mod_batch]
            input_ids, token_type_ids, label_ids, attention_mask = inp

            # obtain loss
            model_out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=label_ids
            )

            loss = model_out['loss']

            # reshape model outputs in shape of evaluation
            if train_step % display_freq == 0:
                with torch.no_grad():
                    logits = model_out['logits']
                    prob = logits.detach().cpu().numpy()
                    pred = (torch.argmax(logits, dim=2)).detach().cpu().numpy()
                    doc = meta_data_batch[1]
                    gt = meta_data_batch[0]
                    doc, gt, pred, pairs = display_result_present(
                        doc_text=doc,
                        doc_pred=pred,
                        doc_prob=prob,
                        doc_gt=gt,
                        inv_label_map=inv_label_map)
                    display_str = f'\nDocument Text: {doc}\n Ground Truth KP: {gt}\n Predicted KP: {pred}\n'
                    print(display_str)

            # loss from multigpu, compute mean
            # ToDO: add num_gpu check
            loss = loss.mean()

            # update bar
            train_bar.set_description(
                'Iter (loss=%5.3f)' % loss.item()
            )
            # model update after every step (batch)
            loss.backward()
            # use scheduler updated lr
            # calling optimizer.step() before scheduler.step() following PyTorch 1.1 recomm.
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # validate model
        if kwargs['args']['local_rank'] == 0:
            # validate model
            result = pke_base_validater(
                val_loader=val_loader,
                model=model,
                device=device,
                inv_label_map=inv_label_map,
                top_k=val_conf['top_k'],
                thresh=val_conf['thresh'],
                save_dir=save_dir
            )

            # save checkpoint
            save_checkpoint(
                save_dir=save_dir,
                model=model,
                epoch=ep
            )


def pke_base_validater(
        val_loader,
        model,
        device,
        inv_label_map,
        top_k,
        thresh,
        save_dir,
        **kwargs
):
    ##### EVALUATION #####
    val_bar = tqdm(
        val_loader,
        desc='Iter (X)',
        disable=False
    )
    model.eval()
    with torch.no_grad():
            # store results across all steps (batches)
        prob_list, pred_list, doc_list, gt_list = [], [], [], []
        for val_step, batch in enumerate(val_bar):
                # update bar
            val_bar.set_description(
                'Iter:%5d' % val_step
            )

            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:4], batch[4:]
            inp = [tens.to(device) for tens in mod_batch]
            input_ids, token_type_ids, label_ids, attention_mask = inp
            model_out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                # when passing None, model returns logits instead of loss (validation)
                labels=None
            )
            logits = model_out['logits']
            # reshape model outputs in shape of evaluation
            # extend reduces the flattens the batch size and adds slices removing batch dim
            prob_list.extend(logits.detach().cpu().numpy())
            # pred_list.extend((torch.argmax(logits, dim=2)).detach().cpu().numpy())
            doc_list.extend(meta_data_batch[1])
            gt_list.extend(meta_data_batch[0])

        
        res, preds, significance_scores = evaluate_present(
            prob_list=prob_list,
            doc_list=doc_list,
            gt_list=gt_list,
            inv_label_map=inv_label_map,
            top_k=top_k,
            thresh=thresh
        )

        # write results
        pred_path = save_dir + f'_predictions.txt'
        with open(pred_path, 'w') as f:
            for line in preds:
                f.write(line + '\n')

        print(f'\nResults: ', res)
        return res, significance_scores


# absent key phrase trainer
# encoder decoder model
def akp_base_trainer(
        model,
        train_loader,
        val_loader,
        epochs,
        device,
        optimizer,
        tokenizer,
        scheduler,
        display_freq,
        val_conf,
        save_dir,
        **kwargs
):
    for ep in trange(0,
                     epochs,
                     desc="Epoch",
                     disable=kwargs['args']['local_rank'] != 0):
        #### TRAIN #####
        train_bar = tqdm(
            train_loader,
            desc='Iter (loss=X.XXX)',
            disable=kwargs['args']['local_rank'] != 0
        )
        model.train()
        for train_step, batch in enumerate(train_bar):
            model.module.config.istrain = True
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:8], batch[8:]
            inp = [tens.to(device) for tens in mod_batch]
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, cross_attention_mask, labels, encoder_mlm_labels_mask, _ = inp

            # obtain loss
            model_out = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              cross_attention_mask=cross_attention_mask,
                              decoder_input_ids=decoder_input_ids,
                              decoder_attention_mask=decoder_attention_mask,
                              labels=labels)

            # ignore mlm and phloss
            loss, _, _ = model_out.loss

            # reshape model outputs in shape of evaluation
            if train_step % display_freq == 0 and kwargs['args']['local_rank'] == 0:
                with torch.no_grad():
                    # logits = model_out['logits']
                    # list of predicted sequence using BEAM Search or other strategies
                    # model.module.generate if using DataParallel else model.generate
                    pred = tokenizer.batch_decode(model.module.generate(
                        input_ids, cross_attention_mask=cross_attention_mask), skip_special_tokens=True)
                    doc = meta_data_batch[0]
                    ab_kp = meta_data_batch[1]

                    doc, gt, pred = display_result_absent(
                        doc_text=doc,
                        doc_pred=pred,
                        doc_gt=ab_kp)

                    display_str = f'\nDocument Text: {doc}\n Ground Truth AKP: {gt}\n Generate AKP: {pred}\n'
                    print(display_str)

            # loss from multigpu, compute mean
            # ToDO: add num_gpu check
            loss = loss.mean()

            # update bar
            train_bar.set_description(
                'Iter (loss=%5.3f)' % loss.item()
            )
            # model update after every step (batch)
            loss.backward()
            # use scheduler updated lr
            # calling optimizer.step() before scheduler.step() following PyTorch 1.1 recomm.
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # validate model
        if kwargs['args']['local_rank'] == 0:
            result = akp_base_validater(
                val_loader=val_loader,
                model=model,
                device=device,
                top_k=val_conf['top_k'],
                tokenizer=tokenizer,
                save_dir=save_dir
            )

            # save checkpoint
            save_checkpoint(
                save_dir=save_dir,
                model=model,
                epoch=ep
            )

# absent key phrase evaluator


def akp_base_validater(
        val_loader,
        model,
        device,
        top_k,
        tokenizer,
        save_dir,
        **kwargs
):
    ##### EVALUATION #####
    val_bar = tqdm(
        val_loader,
        desc='Iter (X)',
        disable=False
    )
    model.eval()
    with torch.no_grad():
            # store results across all steps (batches)
        pred_list, doc_list, gt_list = [], [], []
        for val_step, batch in enumerate(val_bar):
                # update bar
            val_bar.set_description(
                'Iter:%5d' % val_step
            )
            model.module.config.istrain = False
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:8], batch[8:]
            inp = [tens.to(device) for tens in mod_batch]
            input_ids, _, _, _, cross_attention_mask, _, _, _ = inp
            # attention mask is prepared in .generate using inputs.ne(pad_token_id).long()
            # ToDO: can pass it manually as part of **kwargs to the .generate() method
            # pred is a list of list of absent key phrases for each document in batch
            pred = tokenizer.batch_decode(model.module.generate(
                input_ids, cross_attention_mask=cross_attention_mask), skip_special_tokens=True)
            # reshape model outputs in shape of evaluation
            # extend reduces the flattens the batch size and adds slices removing batch dim
            pred_list.extend(pred)
            doc_list.extend(meta_data_batch[0])
            gt_list.extend(meta_data_batch[1])

        res, preds, _, significance_scores = evaluate_absent(
            doc_list=doc_list,
            gt_list=gt_list,
            pred_list=pred_list,
            top_k=top_k
        )

        print(f'\nResults: ', res)

        # write results
        pred_path = save_dir + f'_predictions.txt'
        with open(pred_path, 'w') as f:
            for line in preds:
                f.write(line + '\n')

        return res, significance_scores


# absent key phrase seq2set trainer
def akp_seq2set_trainer(
        model,
        train_loader,
        val_loader,
        epochs,
        device,
        optimizer,
        tokenizer,
        scheduler,
        display_freq,
        val_conf,
        save_dir,
        **kwargs
):
    # Define logits processor list
    custom_logits_processor_list = LogitsProcessorList([
        UnitLogitsProcessor(max_kp_len=model.module.config.max_kp_len,
                            cls_token_id=[tokenizer.cls_token_id]),
    ])

    for ep in trange(0,
                     epochs,
                     desc="Epoch",
                     disable=kwargs['args']['local_rank'] != 0):
        torch.cuda.empty_cache()
        #### TRAIN #####
        train_bar = tqdm(
            train_loader,
            desc='Iter (loss=X.XXX, mlm_loss=X.XXX, , ph_loss=X.XXX, total_loss=X.XXX)',
            disable=kwargs['args']['local_rank'] != 0
        )
        model.train()
        for train_step, batch in enumerate(train_bar):
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:10], batch[-2:]
            # mod_batch = mod_batch+loss_weights
            inp = [tens.to(device) for tens in mod_batch]
            input_ids, attention_mask, decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_attention_mask, cross_attention_mask, labels, encoder_mlm_labels_mask, loss_weights = inp
            # print('\nDDP: inputid shape: ', input_ids.shape)
            # obtain loss
            model.module.config.istrain = True
            # with model.no_sync():
            
            model_out = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              decoder_input_ids=decoder_input_ids,
                              decoder_position_ids=decoder_position_ids,
                              decoder_token_type_ids=decoder_token_type_ids,
                              decoder_attention_mask=decoder_attention_mask,
                              cross_attention_mask=cross_attention_mask,
                              labels=labels,
                              loss_weight_mask=loss_weights,
                              encoder_labels_mask=encoder_mlm_labels_mask)

            loss, mlm_loss, ph_loss = model_out.loss

            # reshape model outputs in shape of evaluation
            # commenting train display for testing..
            if train_step % display_freq == 0 and kwargs['args']['local_rank'] == 0:
                print('\n\nTrain Display')

                with torch.no_grad():
                    # logits = model_out['logits']
                    # list of predicted sequence using BEAM Search or other strategies
                    # model.module.generate if using DataParallel else model.generate
                    model.module.config.istrain = False
                    # cross_attention_mask = cross_attention_mask.repeat_interleave(
                    #     model.module.config.max_kps, dim=0)

                    # Note: only works if val_batch_size is in multiples less than train batch size in the config
                    bs, max_kps = val_conf['batch_size_val'], model.module.config.max_kps
                    # reduce input_ids batch size to batch_size_val
                    gen_input_ids = input_ids[:val_conf['batch_size_val'], :]
                    gen_cross_attention_mask = cross_attention_mask[:val_conf['batch_size_val'], :]
                    gen_decoder_position_ids = decoder_position_ids[:val_conf['batch_size_val'], :]
                    gen_decoder_token_type_ids = decoder_token_type_ids[:val_conf['batch_size_val'], :]

                    gen_decoder_input_ids = decoder_input_ids.new_zeros(
                        bs*max_kps, 1)
                    gen_decoder_input_ids[:, 0] = 101
                    gen_decoder_position_ids = gen_decoder_position_ids.contiguous().view(bs*max_kps, -1)

                    # explicitly pass expanded decoder input_ids
                    # inputs represent encoder input ids (encoder_decoder_model)
                    # print('bs, max_kps: ', bs, max_kps)
                    
                    generation_res = model.module.generate(inputs=gen_input_ids,
                                                   return_dict_in_generate=True,
                                                   output_scores=True,
                                                   cross_attention_mask=gen_cross_attention_mask,
                                                   decoder_input_ids=gen_decoder_input_ids,
                                                   decoder_token_type_ids=gen_decoder_token_type_ids,
                                                   decoder_position_ids=gen_decoder_position_ids,
                                                   output_hidden_states=True)
                    # (bs, max_kps*max_kp_len)
                    # generation_res = model.module.unit_generation_past_kv(
                    #     input_ids=input_ids,
                    #     attention_mask=attention_mask,
                    #     cross_attention_mask=cross_attention_mask,
                    #     decoder_input_ids=decoder_input_ids
                    # )
                    # unstack response for decoding with bert generation library (bs*max_kps, max_kp_len) -> (bs, max_kps, max_kp_len)
                    _, seq_len = generation_res.sequences.size()
                    # print('generation_res (in train): ', generation_res.sequences.size())

                    generation_res.sequences = generation_res.sequences.contiguous().view(-1, model.module.config.max_kps, seq_len).view(
                        -1, model.module.config.max_kps*seq_len
                    )

                    assert generation_res.sequences.shape[-1] == (model.module.config.max_kps *
                                                                  seq_len), \
                        f'generated output shape mismatch {generation_res.sequences.shape}, input_ids {decoder_input_ids.shape}'
                    # skip first [CLS] output token by default
                    # add first additional [CLS] token to end to sequence to make length divisible
                    generation_res = torch.cat(
                        [generation_res.sequences[:, 1:], generation_res.sequences[:, 0:1]], dim=-1)
                    gen_seq = generation_res.contiguous().view(-1, model.module.config.max_kps,
                                                               seq_len)
                    # replace last token of every unit with token corresponding to akp seperator (here ;)
                    gen_seq[:, :, -1] = tokenizer.convert_tokens_to_ids(';')
                    gen_seq = gen_seq.contiguous().view(-1, model.module.config.max_kps *
                                                        seq_len)

                    # skipping special tokens as end of units replaced with ; seperator
                    # which can be used to split the key phrases
                    post_pred = tokenizer.batch_decode(
                        gen_seq, skip_special_tokens=False)

                    doc = meta_data_batch[0][:val_conf['batch_size_val']]
                    ab_kp = meta_data_batch[1][:val_conf['batch_size_val']]

                    doc, gt, pred = display_result_absent(
                        doc_text=doc,
                        doc_pred=post_pred,
                        doc_gt=ab_kp)

                    display_str = f'\nDocument Text: {doc}\n Ground Truth AKP: {gt}\n Generate AKP: {pred}\n'
                    print(display_str)
                # exit(0)

            # loss from multigpu, compute mean
            # ToDO: add num_gpu check
            loss = loss.mean()
            combined_loss = loss
            if mlm_loss:
                mlm_loss = mlm_loss.mean()
                combined_loss = loss + mlm_loss
            if ph_loss:
                ph_loss = ph_loss.mean()
                combined_loss = combined_loss + ph_loss

            # update bar
            train_bar.set_description(
                'Iter (loss=%5.3f, mlm_loss=%5.3f, ph_loss=%5.3f, total_loss=%5.3f)' % (
                    loss.item(), mlm_loss.item() if mlm_loss else 0.0, ph_loss.item() if ph_loss else 0.0, combined_loss.item())
            )
            # model update after every step (batch)
            # update loss of individual components indepedantly
            # loss for mlm
            #     if mlm_loss:
            #         mlm_loss.backward(retain_graph = True)
            # loss.backward()

            combined_loss.backward()

            # use scheduler updated lr
            # calling optimizer.step() before scheduler.step() following PyTorch 1.1 recomm.
            # after loss backward first clip the gradients to prevent gradient explosion problem
            model_params = [p for p in model.parameters() if p.requires_grad]
            # print('\n Length that requires grad: ',len(model_params))
            torch.nn.utils.clip_grad_norm_(model_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # print current learning rate for both param groups
            # print('\n Current learning rate: ')
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])


        print('\nValidating...')
        # validate model
        if kwargs['args']['local_rank'] == 0:
            # result, preds = akp_seq2set_validater(
            #     val_loader=val_loader,
            #     model=model,
            #     device=device,
            #     top_k=val_conf['top_k'],
            #     tokenizer=tokenizer,
            #     save_dir=save_dir
            # )

            # save checkpoint
            save_checkpoint(
                save_dir=save_dir,
                model=model,
                epoch=ep
            )
            # reduce memory fragmentation
            torch.cuda.empty_cache()


# absent key phrase evaluator for seq2set
def akp_seq2set_validater(
        val_loader,
        model,
        device,
        top_k,
        tokenizer,
        save_dir,
        **kwargs
):
    # Define logits processor list
    custom_logits_processor_list = LogitsProcessorList([
        UnitLogitsProcessor(max_kp_len=model.module.config.max_kp_len,
                            cls_token_id=[tokenizer.cls_token_id]),
    ])

    ##### EVALUATION #####
    val_bar = tqdm(
        val_loader,
        desc='Iter (X)',
        disable=False
    )
    model.eval()
    with torch.no_grad():
            # store results across all steps (batches)
        pred_list, doc_list, gt_list = [], [], []
        dec_hs_list = []
        # sorted_scores_list = []
        # sorted_dhead_key_list = []
        # sorted_dhead_value_list = []
        sorted_dec_hs_list = []
        precision_head_mask = []
        for val_step, batch in enumerate(val_bar):
                # update bar
            val_bar.set_description(
                'Iter:%5d' % val_step
            )

            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:10], batch[-2:]
            # mod_batch = mod_batch+loss_weights
            inp = [tens.to(device) for tens in mod_batch]
            # input_ids, _, _, _, cross_attention_mask, _, _, _ = inp
            input_ids, attention_mask, decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_attention_mask, cross_attention_mask, _, _, _ = inp

            # attention mask is prepared in .generate using inputs.ne(pad_token_id).long()
            # ToDO: can pass it manually as part of **kwargs to the .generate() method
            # pred is a list of list of absent key phrases for each document in batch
            model.module.config.istrain = False
            # cross_attention_mask = cross_attention_mask.repeat_interleave(
            #     model.module.config.max_kps, dim=0)

            bs, max_kps = input_ids.size(
            )[0], model.module.config.max_kps

            gen_decoder_input_ids = decoder_input_ids.new_zeros(
                bs*max_kps, 1)
            gen_decoder_input_ids[:, 0] = 101
            gen_decoder_position_ids = decoder_position_ids.contiguous().view(bs*max_kps, -1)
            gen_decoder_decoder_token_type_ids = decoder_token_type_ids.contiguous().view(bs*max_kps, -1)
            
            # print('get_decoder_input_ids_shape: ', gen_decoder_input_ids.shape,
            #       'input ids size: ', input_ids.size(), bs, max_kps)

            # by default use_cache is taken from generation config
            generation_res = model.module.generate(inputs=input_ids,
                                                   return_dict_in_generate=True,
                                                   output_scores=True,
                                                   cross_attention_mask=cross_attention_mask,
                                                   decoder_input_ids=gen_decoder_input_ids,
                                                   decoder_position_ids=gen_decoder_position_ids,
                                                   decoder_token_type_ids=gen_decoder_decoder_token_type_ids,
                                                   output_hidden_states=True)
            
            # store decoder_hidden_states
            # generation_res_dec_hs = generation_res.decoder_hidden_states
            # shape will be (seq_len, 13 layers, bs*num_beams, max_kps*max_seq_len, 768)
            # first seq_len, is because of the autoregressive generation, over time steps
            # dec_hs = generation_res_dec_hs[-1][-1]
            # dec_hs_list.append(dec_hs)
            # print(len(dec_hs_list))
            # print('dec_hs: ', dec_hs.shape)

            ###### Custom implementation for Greedy Search Decoder ##########
            # generation_res = model.module.unit_generation_student_forcing(
            #                                             input_ids=input_ids,
            #                                             cross_attention_mask=cross_attention_mask,
            #                                             decoder_input_ids=gen_decoder_input_ids,
            #                                             decoder_position_ids=gen_decoder_position_ids,
            #                                             encoder_outputs=None,
            #                                             decoder_attention_mask=decoder_attention_mask)

            # 32, 5  (bs=4, max_kp:8, seq_len:5)
            _, seq_len = generation_res.sequences.size()
            
            if hasattr(generation_res, 'decoder_precision_head_mask'):
                if generation_res.decoder_precision_head_mask is not None:
                    # processing here
                    # generation res decoder_precision_head_mask has shape (bs*max_kps, 2)
                    sm = torch.nn.Softmax(dim=1)

                    # post processing to add only include precision heads if prob > 90%
                    generation_res.decoder_precision_head_mask = generation_res.decoder_precision_head_mask.contiguous().view(-1, 2)
                    prec_head_conf_thresh = 0.50
                    prec_hm_sm = sm(generation_res.decoder_precision_head_mask)
                    # bs*max_kps, 2 -> bs*max_kps, 2

                    # print('\n prec_hm_sm shape: ', prec_hm_sm.shape)
                    # print('\n old prec hm sm: ', prec_hm_sm[:,0])
                    prec_hm_sm_ = torch.where(prec_hm_sm[:, :1] < prec_head_conf_thresh, prec_hm_sm.new_zeros(
                        prec_hm_sm[:, :1].size()), prec_hm_sm[:, :1])
                    
                    prec_hm_sm = torch.cat(
                        [prec_hm_sm_, prec_hm_sm[:, 1:]], dim=-1)
                    # print('\n prec_hm_sm: ', prec_hm_sm.shape)
                    # print('\n new prec hm sm: ', prec_hm_sm)
                    prec_hm = prec_hm_sm.argmax(dim=-1)
                    prec_hm = prec_hm.contiguous().view(-1, max_kps)
                    sorted_prec_hm = prec_hm

                    print('sorted prec hm: ', sorted_prec_hm)
                else:
                    sorted_prec_hm = []

            # score shape = (bs*max_kp, vocab_size) , seq_len-1 such tuples

            # sort them based on indices and gather

            # custom decoding function (for greedy beam search decoding: not supported by Huggingface library)
            # generation_res = model.module.beam_search_decode(input_ids=input_ids,
            #                                                  attention_mask=attention_mask,
            #                                                  cross_attention_mask=cross_attention_mask,
            #                                                  decoder_input_ids=gen_decoder_input_ids,
            #                                                  decoder_attention_mask=decoder_attention_mask,
            #                                                  encoder_outputs=None)
            # _, seq_len = generation_res.sequences.size()

            max_kps, return_seq = model.module.config.max_kps, model.module.config.num_return_sequences
            kp_per_dp = return_seq*max_kps

            ####SORT for top-k#####
            # For Beam Search Decoding
            if hasattr(generation_res, 'sequences_scores'):
                sequences_scores = generation_res.sequences_scores
                B = sequences_scores.size()[0]
                bs = B//(max_kps*return_seq)
                # print('kp per dp: ', kp_per_dp)

                sequences_scores = sequences_scores.view(-1, kp_per_dp)
                sorted_indices = sequences_scores.argsort(
                    dim=1, descending=True)
                sorted_scores = sequences_scores.gather(1, sorted_indices)
                # print('sorted indices: ', sorted_indices)
                # print('sequence scores: ', sequences_scores)
                # print('sorted_indices', sorted_indices, sorted_indices.shape, generation_res.sequences.shape)
                sorted_sequences = generation_res.sequences.view(-1, kp_per_dp, seq_len).gather(
                    1, sorted_indices.unsqueeze(-1).expand(-1, -1, seq_len))
                # print('sorted sequences shape: ', sorted_sequences.shape)
                generation_res.sequences = sorted_sequences.view(-1, seq_len)

                # sort prec head mask if its there and sorted indices is available
                # otherwise if precision head mask is available use the unsorted one
                if hasattr(generation_res, 'decoder_precision_head_mask'):
                    if generation_res.decoder_precision_head_mask is not None:
                        # print('\n before sorting precision hm: \n', prec_hm, '\n', sorted_indices)
                        sorted_prec_hm = prec_hm.gather(1, sorted_indices)

            # For Greedy Search Decoding
            elif hasattr(generation_res, 'scores'):
                    # print('sorting failed..')
                    # print('\nGreedy generation scores shape: ', len(generation_res.scores), generation_res.scores[0].shape, generation_res.scores[0].shape)
                sequences_scores = torch.stack(
                    list(generation_res.scores), dim=1)
                # scores = scores.max(dim=-1).values
                # print(scores.shape) # bs*max_kps, seq_len
                sequences_scores = sequences_scores.max(
                    dim=-1).values.sum(dim=-1)  # bs*max_kps, seq_len -> bs*max_kps
                # print('scores shape: ', scores.shape)
                sequences_scores = sequences_scores.contiguous().view(-1, kp_per_dp)
                sorted_indices = sequences_scores.argsort(
                    dim=1, descending=True)
                sorted_scores = sequences_scores.gather(1, sorted_indices)
                # print('sorted_scores: ', sorted_scores, sorted_scores.shape)
                # print('sorted indices shape: ',sorted_indices.shape)
                sorted_sequences = generation_res.sequences.view(-1, kp_per_dp, seq_len).gather(
                    1, sorted_indices.unsqueeze(-1).expand(-1, -1, seq_len)
                )
                generation_res.sequences = sorted_sequences.view(-1, seq_len)

                # sort prec head mask if its there and sorted indices is available
                # otherwise if precision head mask is available use the unsorted one
                if hasattr(generation_res, 'decoder_precision_head_mask'):
                    if generation_res.decoder_precision_head_mask is not None:
                        # print('\n before sorting precision hm: \n', prec_hm, '\n', sorted_indices)
                        sorted_prec_hm = prec_hm.gather(1, sorted_indices)
            ##################

            # generation_res.sequences = generation_res.sequences.view(-1, seq_len)
            sorted_prec_hm = []
            precision_head_mask.extend(sorted_prec_hm)

            generation_res = torch.cat(
                [generation_res.sequences[:, 1:], generation_res.sequences[:, 0:1]], dim=-1)
            # 64, 6     2, 32, 6
            gen_seq = generation_res.contiguous().view(-1, kp_per_dp,
                                                       seq_len)

            # def display(x):
            #     for y in x:
            #         print(y, '\n')

            # post processing to replace any token after SEP token to SEP
            # print('\n\n Before sep post processing: \n')
            # display(tokenizer.batch_decode(
            #     gen_seq.contiguous().view(-1, kp_per_dp *
            #                                     seq_len), skip_special_tokens=False))
            
            pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
            none_token_id = tokenizer.convert_tokens_to_ids('none')
            # print('none token id: ', none_token_id)
            sep_positions = (gen_seq == sep_token_id) | (gen_seq == none_token_id)
            il, jl, kl = gen_seq.size()
            for i in range(il):
                for j in range(jl):
                    flag = 0
                    for k in range(kl):
                        if sep_positions[i][j][k]:
                            flag = 1
                        if flag:
                            gen_seq[i][j][k] = pad_token_id
            # print('\n\n After sep post processing: \n')
            # display(tokenizer.batch_decode(
            #     gen_seq.contiguous().view(-1, kp_per_dp *
            #                                     seq_len), skip_special_tokens=False))
            ####
            # replace last token of every unit with token corresponding to akp seperator (here ;)
            gen_seq[:, :, -1] = tokenizer.convert_tokens_to_ids(';')

            gen_seq = gen_seq.contiguous().view(-1, kp_per_dp *
                                                seq_len)
            # print('gen seq: ', gen_seq.shape)
            # skipping special tokens as end of units replaced with ; seperator
            # which can be used to split the key phrases
            post_pred = tokenizer.batch_decode(
                gen_seq, skip_special_tokens=True)
            # preserve prediction head information
            #

            # print(f'\n post pred len: {len(post_pred)}')
            # for x in post_pred:
            #     print(x)
            # print('\n appended separator')

            # reshape model outputs in shape of evaluation
            # extend reduces the flattens the batch size and adds slices removing batch dim
            pred_list.extend(post_pred)
            doc_list.extend(meta_data_batch[0])
            gt_list.extend(meta_data_batch[1])

            # doc = meta_data_batch[0]
            # ab_kp = meta_data_batch[1]
            # doc, gt, pred = display_result_absent(
            #             doc_text=doc,
            #             doc_pred=post_pred,
            #             doc_gt=ab_kp)


        res, preds, dhead_data, significance_scores = evaluate_absent(
            doc_list=doc_list,
            gt_list=gt_list,
            pred_list=pred_list,
            top_k=top_k
        )

        print(f'\nResults', res)

        # write results
        pred_path = save_dir + f'_predictions.txt'
        with open(pred_path, 'w') as f:
            for line in preds:
                f.write(line + '\n')

        # get all head predictions
        res_units, preds_units = evaluate_absent_all_heads(
            doc_list=doc_list,
            gt_list=gt_list,
            pred_list=pred_list,
            top_k=top_k,
            # sequences_scores=sorted_scores_list,
            # hidden_states_sorted=sorted_dec_hs_list,
            precision_head_mask=precision_head_mask
        )

        # save predictions for each unit
        for unit_ind, pred_unit in enumerate(preds_units):
            pred_path = save_dir + f'_predictions_{unit_ind}.txt'
            with open(pred_path, 'w') as f:
                for line in pred_unit:
                    f.write(line + '\n')

        # save the hidden states 
        # import pickle
        # with open(pred_path[:-4]+"_hiddenstates.pickle", "wb") as f:
        #     pickle.dump(dec_hs_list, f)

        return res, res_units, significance_scores


# proposed method emb2set trainer
def akp_emb2set_trainer(
        model,
        train_loader,
        val_loader,
        epochs,
        device,
        optimizer,
        tokenizer,
        scheduler,
        display_freq,
        val_conf,
        run_conf,
        save_dir,
        **kwargs
):

    for ep in trange(0,
                     epochs,
                     desc="Epoch",
                     disable=kwargs['args']['local_rank'] != 0):
        #### TRAIN #####
        train_bar = tqdm(
            train_loader,
            desc='Iter (loss=X.XXX)',
            disable=kwargs['args']['local_rank'] != 0
        )
        model.train()
        vanilla_NAR = model.module.config.vanilla_nar
        inp_embeds = model.module.config.input_embeds

        for train_step, batch in enumerate(train_bar):
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:7], batch[7:]

            inp = [tens.to(device) for tens in mod_batch]
            input_ids, token_attention_mask, input_embeds, embeds_attention_mask, labels, encoder_hidden_states, encoder_attention_mask = inp

            if vanilla_NAR:
                model_out = model(input_ids=input_ids,
                                  attention_mask=token_attention_mask,
                                  labels=labels,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
            elif inp_embeds:
                model_out = model(inputs_embeds=input_embeds,
                                  attention_mask=embeds_attention_mask,
                                  labels=labels,
                                  encoder_hidden_states=None,
                                  encoder_attention_mask=None)
            else:
                model_out = model(input_ids=input_ids,
                                  attention_mask=token_attention_mask,
                                  labels=labels,
                                  encoder_hidden_states=None,
                                  encoder_attention_mask=None)

            loss = model_out.loss

            # reshape model outputs in shape of evaluation
            if train_step % display_freq == 0 and kwargs['args']['local_rank'] == 0:
                with torch.no_grad():
                    # logits = model_out['logits']
                    # list of predicted sequence using BEAM Search or other strategies
                    # model.module.generate if using DataParallel else model.generate

                    doc = meta_data_batch[0]
                    ab_kp = meta_data_batch[1]
                    logits = model_out.logits

                    pred = torch.argmax(logits, dim=2).detach().cpu(
                    ).view(-1, run_conf['max_kps'], run_conf['max_kp_len'])
                    bs, _, _ = pred.size()
                    separator = torch.tensor([tokenizer.convert_tokens_to_ids(';')]).expand(
                        bs, run_conf['max_kps'], 1)
                    post_pred = torch.cat(
                        [pred, separator], dim=-1).view(-1, run_conf['max_kps']*(run_conf['max_kp_len']+1))

                    # convert to tokens
                    post_pred = tokenizer.batch_decode(
                        post_pred, skip_special_tokens=False)

                    doc, gt, pred = display_result_absent(
                        doc_text=doc,
                        doc_pred=post_pred,
                        doc_gt=ab_kp)

                    display_str = f'\nDocument Text: {doc}\n Ground Truth AKP: {gt}\n Generate AKP: {pred}\n'
                    print(display_str)

            # loss from multigpu, compute mean
            # ToDO: add num_gpu check
            loss = loss.mean()

            # update bar
            train_bar.set_description(
                'Iter (loss=%5.3f)' % loss.item()
            )
            # model update after every step (batch)
            loss.backward()
            # use scheduler updated lr
            # calling optimizer.step() before scheduler.step() following PyTorch 1.1 recomm.
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # validate model
        if kwargs['args']['local_rank'] == 0:
            result = akp_emb2set_validater(
                val_loader=val_loader,
                model=model,
                device=device,
                top_k=val_conf['top_k'],
                run_conf=run_conf,
                tokenizer=tokenizer
            )

            # save checkpoint
            save_checkpoint(
                save_dir=save_dir,
                model=model,
                epoch=ep
            )


def akp_emb2set_validater(
        val_loader,
        model,
        device,
        top_k,
        run_conf,
        tokenizer,
        **kwargs
):

    ##### EVALUATION #####
    val_bar = tqdm(
        val_loader,
        desc='Iter (X)',
        disable=False
    )
    model.eval()
    vanilla_NAR = model.module.config.vanilla_nar
    inp_embeds = model.module.config.input_embeds

    with torch.no_grad():
            # store results across all steps (batches)
        pred_list, doc_list, gt_list = [], [], []
        for val_step, batch in enumerate(val_bar):
                # update bar
            val_bar.set_description(
                'Iter:%5d' % val_step
            )

            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:7], batch[7:]

            inp = [tens.to(device) for tens in mod_batch]
            input_ids, token_attention_mask, input_embeds, embeds_attention_mask, labels, encoder_hidden_states, encoder_attention_mask = inp

            if vanilla_NAR:
                model_out = model(input_ids=input_ids,
                                  attention_mask=token_attention_mask,
                                  labels=labels,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
            elif inp_embeds:
                model_out = model(inputs_embeds=input_embeds,
                                  attention_mask=embeds_attention_mask,
                                  labels=labels,
                                  encoder_hidden_states=None,
                                  encoder_attention_mask=None)
            else:
                model_out = model(input_ids=input_ids,
                                  attention_mask=token_attention_mask,
                                  labels=labels,
                                  encoder_hidden_states=None,
                                  encoder_attention_mask=None)

            logits = model_out.logits

            pred = torch.argmax(logits, dim=2).detach().cpu(
            ).view(-1, run_conf['max_kps'], run_conf['max_kp_len'])
            bs, _, _ = pred.size()
            separator = torch.tensor([tokenizer.convert_tokens_to_ids(';')]).expand(
                bs, run_conf['max_kps'], 1)
            post_pred = torch.cat(
                [pred, separator], dim=-1).view(-1, run_conf['max_kps']*(run_conf['max_kp_len']+1))

            # skipping special tokens as end of units replaced with ; seperator
            # which can be used to split the key phrases
            post_pred = tokenizer.batch_decode(
                post_pred, skip_special_tokens=True)

            # reshape model outputs in shape of evaluation
            # extend reduces the flattens the batch size and adds slices removing batch dim
            pred_list.extend(post_pred)
            doc_list.extend(meta_data_batch[0])
            gt_list.extend(meta_data_batch[1])

        res = evaluate_absent(
            doc_list=doc_list,
            gt_list=gt_list,
            pred_list=pred_list,
            top_k=top_k
        )

        print(f'\nResults', res)
        return res

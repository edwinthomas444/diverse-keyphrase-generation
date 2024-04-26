import torch
from tqdm import trange, tqdm
from evaluation.absent_kp_eval import evaluate_absent, display_result_absent
from trainers.base_trainers import save_checkpoint


def akp_exp_trainer(
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
        for train_step, batch in enumerate(train_bar):
            model.module.config.istrain = True
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:4], batch[4:]
            inp = [tens.to(device) for tens in mod_batch]
            input_embeds, embeds_attention_mask, target_labels, target_embeds = inp

            # obtain loss
            model_out = model(input_embeds=input_embeds,
                              attention_mask=embeds_attention_mask,
                              labels=target_labels,
                              labels_embeds=target_embeds,
                              loss_form='distance',
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
            result = akp_exp_validater(
                val_loader=val_loader,
                model=model,
                device=device,
                top_k=val_conf['top_k'],
                tokenizer=tokenizer,
                run_conf=run_conf
            )

            # save checkpoint
            save_checkpoint(
                save_dir=save_dir,
                model=model,
                epoch=ep
            )


def akp_exp_validater(
        val_loader,
        model,
        device,
        top_k,
        tokenizer,
        run_conf,
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
            # move tensors to the device (inp_ids, token_type_ids, label_ids, attention_mask)
            mod_batch, meta_data_batch = batch[:4], batch[4:]
            inp = [tens.to(device) for tens in mod_batch]
            input_embeds, embeds_attention_mask, _, _ = inp

            # obtain loss
            model_out = model(input_embeds=input_embeds,
                              attention_mask=embeds_attention_mask,
                              labels=None,
                              labels_embeds=None,
                              loss_form='distance',
                              encoder_hidden_states=None,
                              encoder_attention_mask=None)

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
            
            post_pred = tokenizer.batch_decode(
                        post_pred, skip_special_tokens=True)
            # reshape model outputs in shape of evaluation
            # extend reduces the flattens the batch size and adds slices removing batch dim
            pred_list.extend(post_pred)
            doc_list.extend(doc)
            gt_list.extend(ab_kp)

        res = evaluate_absent(
            doc_list=doc_list,
            gt_list=gt_list,
            pred_list=pred_list,
            top_k=top_k
        )

        print(f'\nResults: ', res)
        return res

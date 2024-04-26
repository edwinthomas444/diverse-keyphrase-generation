import torch

# collator for present key phrase


def pke_base_collator(batch):
    batch_t = [[] for _ in range(6)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        batch_t[0].append(torch.tensor(belem['input_ids'], dtype=torch.long))
        batch_t[1].append(torch.tensor(
            belem['token_type_ids'], dtype=torch.long))
        batch_t[2].append(torch.tensor(belem['label_ids'], dtype=torch.long))
        batch_t[3].append(torch.tensor(
            belem['attention_mask'], dtype=torch.long))
        batch_t[4].append(belem['gold_kps'])
        batch_t[5].append(belem['doc'])

    # stack tensors along batch dim
    for i in range(4):
        batch_t[i] = torch.stack(batch_t[i], dim=0)
    return batch_t

# collator for absent key phrase


def akp_base_collator(batch):
    batch_t = [[] for _ in range(10)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        batch_t[0].append(torch.tensor(belem['input_ids'], dtype=torch.long))
        batch_t[1].append(torch.tensor(
            belem['attention_mask'], dtype=torch.long))
        batch_t[2].append(torch.tensor(
            belem['decoder_input_ids'], dtype=torch.long))
        batch_t[3].append(torch.tensor(
            belem['decoder_attention_mask'], dtype=torch.long))
        batch_t[4].append(torch.tensor(
            belem['cross_attention_mask'], dtype=torch.long))
        batch_t[5].append(torch.tensor(
            belem['labels'], dtype=torch.long))
        batch_t[6].append(torch.tensor(belem['encoder_mlm_labels_mask'], dtype=torch.long))
        batch_t[7].append(torch.tensor(belem['loss_weight_mask']))

        batch_t[8].append(belem['doc'])
        batch_t[9].append(belem['gold_akp'])

    # stack tensors along batch dim
    for i in range(8):
        batch_t[i] = torch.stack(batch_t[i], dim=0)

    return batch_t

def akp_seq2set_collator(batch):
    batch_t = [[] for _ in range(12)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        batch_t[0].append(torch.tensor(belem['input_ids'], dtype=torch.long))
        batch_t[1].append(torch.tensor(
            belem['attention_mask'], dtype=torch.long))
        batch_t[2].append(torch.tensor(
            belem['decoder_input_ids'], dtype=torch.long))
        batch_t[3].append(torch.tensor(
            belem['decoder_position_ids'], dtype=torch.long))
        batch_t[4].append(torch.tensor(
            belem['decoder_token_type_ids'], dtype=torch.long))
        batch_t[5].append(torch.tensor(
            belem['decoder_attention_mask'], dtype=torch.long))
        batch_t[6].append(torch.tensor(
            belem['cross_attention_mask'], dtype=torch.long))
        batch_t[7].append(torch.tensor(
            belem['labels'], dtype=torch.long))
        batch_t[8].append(torch.tensor(belem['encoder_mlm_labels_mask'], dtype=torch.long))
        batch_t[9].append(torch.tensor(belem['loss_weight_mask']))

        batch_t[10].append(belem['doc'])
        batch_t[11].append(belem['gold_akp'])

    # stack tensors along batch dim
    for i in range(10):
        batch_t[i] = torch.stack(batch_t[i], dim=0)

    return batch_t

def akp_emb2set_collator(batch):

    batch_t = [[] for _ in range(9)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        # print(belem['decoder_input_ids'][0].device, type(belem['decoder_input_ids']))
        batch_t[0].append(torch.tensor(
            belem['decoder_input_ids'], dtype=torch.long))

        batch_t[1].append(torch.tensor(
            belem['decoder_token_attention_mask'], dtype=torch.long))
        batch_t[2].append(torch.tensor(belem['decoder_input_embeds'].clone(
        ).detach().cpu(), dtype=torch.float32).contiguous().view(-1, 768))
        # print(batch_t[2])

        batch_t[3].append(torch.tensor(
            belem['decoder_attention_mask'], dtype=torch.long))
        batch_t[4].append(torch.tensor(
            belem['labels'], dtype=torch.long))
        batch_t[5].append(torch.tensor(
            belem['encoder_hidden_states'].clone().detach().cpu(), dtype=torch.float32))

        batch_t[6].append(torch.tensor(
            belem['encoder_attention_mask'].clone().detach().cpu(), dtype=torch.long))
        batch_t[7].append(belem['doc'])
        batch_t[8].append(belem['gold_akp'])

    # stack tensors along batch dim
    for i in range(7):
        batch_t[i] = torch.stack(batch_t[i], dim=0)

    return batch_t

def akp_exp_emb2set_collator(batch):

    batch_t = [[] for _ in range(6)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        # print(belem['decoder_input_ids'][0].device, type(belem['decoder_input_ids']))
        batch_t[0].append(torch.tensor(
            belem['ekp_embs'], dtype=torch.float32))
        batch_t[1].append(torch.tensor(
            belem['attention_mask'], dtype=torch.long))
        batch_t[2].append(torch.tensor(
            belem['labels'], dtype=torch.long
        ))
        batch_t[3].append(torch.tensor(
            belem['akp_embs'], dtype=torch.float32))
        
        batch_t[4].append(belem['doc'])
        batch_t[5].append(belem['gold_akp'])

    # stack tensors along batch dim
    for i in range(4):
        batch_t[i] = torch.stack(batch_t[i], dim=0)

    return batch_t

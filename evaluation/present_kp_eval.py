import numpy as np
import torch
import random
import textwrap
wrapper = textwrap.TextWrapper(width=100)


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def stem_norm(container):
    result_set = set()
    remove_space_set = set([w.strip() for w in container])
    for keyphrase in remove_space_set:
        stem_keyphrase_list = []
        for word in keyphrase.split(" "):
            stem_keyphrase_list.append(stemmer.stem(word))
        result_set.add(" ".join(stem_keyphrase_list))
    return result_set


def display_result_present(
    doc_text,
    doc_prob,
    doc_pred,
    doc_gt,
    inv_label_map
):
    # get a random index for display in the batch
    disp_ind = random.randrange(0, len(doc_text))
    doc_text = doc_text[disp_ind][0]
    doc_pred = doc_pred[disp_ind][1:]  # remove first [CLS] token
    doc_prob = doc_prob[disp_ind][1:]  # remove first [CLS] token
    doc_gt = doc_gt[disp_ind]

    pred_kps = []
    doc_text_toks = doc_text.strip().split(" ")
    b_start = -1
    pairs = []
    for token_ind, (pred_ind, doc_token) in enumerate(zip(doc_pred, doc_text_toks)):
        pairs.append([pred_ind, doc_token])
        pred_label = inv_label_map[pred_ind]
        token_score = doc_prob[token_ind][pred_ind]
        if pred_label == 'B':
            # start recording KP
            pred_kps.append([doc_token, token_score, 1])
            b_start = token_ind  # start contiguous check
        elif pred_label in ['X', 'I'] and b_start != -1:
            # need this to be contiguous (only following B)
            pred_kps[-1][0] += doc_token.lstrip(
                "##") if doc_token.startswith("##") else ' '+doc_token
            pred_kps[-1][1] += token_score
            pred_kps[-1][2] += 1  # number of words in the keyphrase
        elif pred_label == 'O' and b_start != -1:
            b_start = -1

    # list of predictions
    pred_kps = set([x[0].strip() for x in pred_kps])
    # list of ground truth
    gt_kps = set([x.strip() for x in doc_gt.keys()])
    # restore article (doc_text)
    rest_doc = doc_text.replace(' ##', '')

    return rest_doc, gt_kps, pred_kps, pairs


def evaluate_present(
    prob_list,
    doc_list,
    gt_list,
    inv_label_map,
    top_k=5,
    eps=1e-08,
    thresh=0.0
):
    topk_pred_kps = 0
    total_gt_kps = 0
    topm_pred_kps = 0
    topo_pred_kps = 0

    hitsm = 0
    hitsk = 0
    hitso = 0
    macro_p, macro_r = [], []
    macro_pk, macro_rk = [], []
    macro_po, macro_ro = [], []

    model_preds = []

    pred_list = []
    # if prob 'O' lesser than thresh, make it zero
    for samp in prob_list:
        seq_len, _ = samp.shape
        for i in range(seq_len):
            if samp[i].argmax(axis=0) == 0 and samp[i][0] < thresh:
                samp[i][0] = 0.0
        pred_list.append(samp.argmax(axis=1))

    for doc_prob, doc_text, doc_pred, doc_gt in zip(prob_list, doc_list, pred_list, gt_list):
        # for each document
        pred_kps = []  # each list in this stores a [KP, KP score, length of KP]

        doc_text_toks = doc_text[0].strip().split(" ")
        # print(len(doc_pred),len(doc_text_toks))

        # IMP: omit the class token
        doc_prob = doc_prob[1:]
        doc_pred = doc_pred[1:]

        b_start = -1
        for token_ind, (pred_ind, doc_token) in enumerate(zip(doc_pred, doc_text_toks)):
            pred_label = inv_label_map[pred_ind]
            token_score = doc_prob[token_ind][pred_ind]
            # pred_label in ['B','I','X'] and b_start == -1:
            if pred_label == 'B':
                # start recording KP
                pred_kps.append([doc_token, token_score, 1])
                b_start = token_ind  # start contiguous check
            elif pred_label in ['X', 'I'] and b_start != -1:
                # need this to be contiguous (only following B)
                pred_kps[-1][0] += doc_token.lstrip(
                    "##") if doc_token.startswith("##") else ' '+doc_token
                pred_kps[-1][1] += token_score
                pred_kps[-1][2] += 1  # number of tokens in the keyphrase
            elif pred_label == 'O' and b_start != -1:
                b_start = -1
        
        gold_preds = set([x.strip() for x in doc_gt.keys()])

        sorted_preds = sorted(pred_kps, key=lambda x: -float(x[1])/x[2])
        topk_preds = set([x[0].strip() for x in sorted_preds[:top_k]])
        topm_preds = set([x[0].strip() for x in pred_kps])
        top_o = len(stem_norm(gold_preds))
        topo_preds = set([x[0].strip() for x in sorted_preds[:top_o]])

        topk_pred_kps += len(topk_preds)
        topm_pred_kps += len(topm_preds)
        topo_pred_kps += len(topo_preds)


        total_gt_kps += len(gold_preds)

        curr_hits_m = len(stem_norm(topm_preds) & stem_norm(gold_preds))
        curr_hits_k = len(stem_norm(topk_preds) & stem_norm(gold_preds))
        curr_hits_o = len(stem_norm(topo_preds) & stem_norm(gold_preds))
        # macro scores
        p = curr_hits_m / (len(topm_preds)+eps)
        r = curr_hits_m / (len(gold_preds)+eps)
        f = (2*p*r)/(p+r+eps)

        p_k = curr_hits_k / (len(topk_preds)+eps)
        r_k = curr_hits_k / (len(gold_preds)+eps)
        f_k = (2*p_k*r_k)/(p_k+r_k+eps)

        p_o = curr_hits_o / (len(topo_preds)+eps)
        r_o = curr_hits_o / (len(gold_preds)+eps)
        f_o = (2*p_o*r_o)/(p_o+r_o+eps)

        macro_p.append(p)
        macro_r.append(r)
        # macro_f1.append(f)
        macro_pk.append(p_k)
        macro_rk.append(r_k)
        # macro_f1k.append(f_k)
        macro_po.append(p_o)
        macro_ro.append(r_o)

        # micro scores
        hitsm += curr_hits_m
        hitsk += curr_hits_k
        hitso += curr_hits_o
        # print('\ngt: ', gold_preds)
        # print('preds: ', topm_preds)

        doc_text = '\n' + \
            '\n'.join(wrapper.wrap(' '.join(doc_text).strip().replace(' ##', '')))
        gt_text = 'Ground Truth: ' + ' ; '.join(list(gold_preds))
        pred_text = 'Pred Text: ' + ' ; '.join(list(topm_preds))
        model_preds.extend([doc_text, gt_text, pred_text])

    # final metrics on all documents
    p = hitsm / (topm_pred_kps+eps)
    r = hitsm / (total_gt_kps+eps)
    f = (2*p*r)/(p+r+eps)
    # print(f'p,r,f @ M: {p}, {r}, {f}')

    p_k = hitsk / (topk_pred_kps+eps)
    r_k = hitsk / (total_gt_kps+eps)
    f_k = (2*p_k*r_k)/(p_k+r_k+eps)

    # print(f'p,r,f @ K: {p_k}, {r_k}, {f_k}')
    p_o = hitso / (topo_pred_kps+eps)
    r_o = hitso / (total_gt_kps+eps)
    f_o = (2*p_o*r_o)/(p_o+r_o+eps)

    res_micro = {
        'p': p,
        'r': r,
        'f': f,
        'p_k': p_k,
        'r_k': r_k,
        'f_k': f_k,
        'p_o': p_o,
        'r_o': r_o,
        'f_o': f_o
    }

    p = sum(macro_p)/len(macro_p)
    r = sum(macro_r)/len(macro_r)
    f = (2*p*r)/(p+r+eps)
    # print(f'p,r,f @ M: {p}, {r}, {f}')

    p_k = sum(macro_pk)/len(macro_pk)
    r_k = sum(macro_rk)/len(macro_rk)
    f_k = (2*p_k*r_k)/(p_k+r_k+eps)

    p_o = sum(macro_po)/len(macro_po)
    r_o = sum(macro_ro)/len(macro_ro)
    f_o = (2*p_o*r_o)/(p_o+r_o+eps)

    res_macro = {
        'p': p,
        'r': r,
        'f': f,
        'p_k': p_k,
        'r_k': r_k,
        'f_k': f_k,
        'p_o': p_o,
        'r_o': r_o,
        'f_o': f_o
    }

    # also pass the individual sample scores for significance testing
    macro_fm_list, macro_fk_list, macro_fo_list = [], [], []
    for sp, sr in zip(macro_p, macro_r):
        sf = 2*sp*sr/(sp+sr+eps)
        macro_fm_list.append(sf)

    for sp, sr in zip(macro_pk, macro_rk):
        sf = 2*sp*sr/(sp+sr+eps)
        macro_fk_list.append(sf)

    for sp, sr in zip(macro_po, macro_ro):
        sf = 2*sp*sr/(sp+sr+eps)
        macro_fo_list.append(sf)

    return (res_micro, res_macro), model_preds, (macro_fm_list, macro_fk_list, macro_fo_list)

from nltk.stem.porter import PorterStemmer
import random
import textwrap
import copy
from sklearn_extra.cluster import KMedoids

wrapper = textwrap.TextWrapper(width=100)

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


def ordered_stem_norm(container):
    result_list = list()
    remove_space_list = list([w.strip() for w in container])
    for keyphrase in remove_space_list:
        stem_keyphrase_list = []
        for word in keyphrase.split(" "):
            stem_keyphrase_list.append(stemmer.stem(word))
        result_list.append(" ".join(stem_keyphrase_list))
    # print('len result list: ', len(result_list), len(container))
    return result_list

# ToDo: need to add probabilites for confidence thresholding of heads (head masking)


def evaluate_absent_all_heads(
        doc_list,
        gt_list,
        pred_list,
        top_k=5,
        eps=1e-08,
        **kwargs
):
    num_units = len(pred_list[0].split(';')) - 1
    # print('\nnum units: ', num_units)

    micro_heads, macro_heads = [], []
    model_preds_heads = []
    for unit_ind in range(num_units):
        # extract predictions of unit head (filter out all other head predictions)
        pred_list_ = [x.split(';')[unit_ind] for x in pred_list]
        # last head use ensemble with custom head selection logic options
        if unit_ind == num_units-2:
            # last but one head gets ensemble of sorted heads based on confidence scores
            # first half of the units
            pred_list_ = [';'.join(x.split(';')[:4]) for x in pred_list]

        elif unit_ind == num_units-1:
            # last head gets ensemble of heads selected by precision head mask
            if 'precision_head_mask' in kwargs.keys() and len(kwargs['precision_head_mask']):
                ph_mask = kwargs['precision_head_mask']
                # list of max_kps len list, outerlist has bs length
                pred_list_ = []
                for ind, preds in enumerate(pred_list):
                    # everything except last appended semicolon seperator
                    kps = preds.split(';')[:-1]
                    mask = ph_mask[ind]
                    # print('\m Mask inference: ', mask)
                    # print('\n kps: ', kps)
                    kps_ = [x for i, x in enumerate(kps) if mask[i] == 1]
                    preds_ = ';'.join(kps_)
                    # print('\n new kps: ', preds_)
                    pred_list_.append(preds_)

            elif 'hidden_states_sorted' in kwargs.keys():
                hidden_states = kwargs['hidden_states_sorted']
                # cluster based on hidden states
                num_clusters = num_units//2
                pred_list_ = []
                for ind, pred_hs in enumerate(hidden_states):
                    # max_kps, 768
                    kmedoids_instance = KMedoids(
                        n_clusters=num_clusters, random_state=42)
                    clusters = kmedoids_instance.fit_predict(pred_hs)
                    # mediods to include (sort the cluster indices for top-k evaluation)
                    mediods_inds = sorted(kmedoids_instance.medoid_indices_)
                    # filter all other other head predictions except the mediods
                    kps = pred_list[ind].split(';')

                    list_ = ';'.join([kps[i] for i in mediods_inds])
                    pred_list_.append(list_)
                    # print('\n mediod indices: ', mediods_inds)
                    # print('\n clusters: ', clusters)

                    # print('\n old kps: ', kps)
                    # print('\n new mediod kps: ', list_)
                    # print('\n gt kps: ',gt_list[ind])
            elif 'sequences_scores' in kwargs.keys():
                pred_list_ = []
                # max_kps per document
                sequences_scores = kwargs['sequences_scores']
                # find location where
                assert len(pred_list) == len(sequences_scores)
                for x, ss in zip(pred_list, sequences_scores):
                    # print('ss>=60', ss, ss>=60)
                    bisect_index = (ss >= 58).sum(dim=0)
                    # print('bisect index: ',bisect_index)
                    pred_list_.append(';'.join(x.split(';')[:bisect_index]))
            else:
                pred_list_ = [';'.join(x.split(';')[:4]) for x in pred_list]

        # print(f'pred_list head {unit_ind}', pred_list_)
        (res_micro, res_macro), model_preds, (_, _), _ = evaluate_absent(doc_list=doc_list,
                                                                         gt_list=gt_list,
                                                                         pred_list=pred_list_,
                                                                         top_k=top_k,
                                                                         eps=eps,
                                                                         **kwargs)
        micro_heads.append(res_micro)
        macro_heads.append(res_macro)
        model_preds_heads.append(model_preds)
    # print('model preds: ', model_preds[-1])

    return (micro_heads, macro_heads), model_preds_heads


def evaluate_absent(
    doc_list,
    gt_list,
    pred_list,
    top_k=5,
    eps=1e-08,
    **kwargs
):
    topk_pred_kps = 0
    topo_pred_kps = 0
    total_gt_kps = 0
    topm_pred_kps = 0
    hitsm = 0
    hitsk = 0
    hitso = 0

    macro_p, macro_r = [], []
    macro_pk, macro_rk = [], []
    macro_po, macro_ro = [], []

    model_preds = []

    # print('\ndoc_list len: ', len(doc_list), 'pred list len: ', len(pred_list), 'gt list len: ', len(gt_list))
    # get the top-5 key phrases
    dhead_gts, dhead_kps = [], []
    for doc_pt, doc_pred, doc_gt in zip(doc_list, pred_list, gt_list):

        # doc pt is the tokens concatenated with space in single list

        # for each document
        # each list in this stores a [KP, KP score, length of KP]

        # get key phrases list (filter empty kps '')
        topm_akp_preds = list(dict.fromkeys(
            [x.strip() for x in doc_pred.split(';') if x]))
        # remove phi tokens ([UNK])
        topm_akp_preds = [x for x in topm_akp_preds if (
            x and x != '[UNK]' and x != 'none')]

        gold_preds = set(doc_gt)

        # remove duplicates and take top-k in order
        topk_akp_preds = topm_akp_preds[:top_k]
        # find top-o kps
        gt_len = len(stem_norm(gold_preds))
        topo_akp_preds = topm_akp_preds[:gt_len]

        topm_akp_preds, topk_akp_preds, topo_akp_preds = set(
            topm_akp_preds), set(topk_akp_preds), set(topo_akp_preds)

        # for meta learner
        raw_akp_preds = [x.strip() for x in doc_pred.split(';')]
        # print('\nraw akp preds: ', raw_akp_preds)
        raw_akp_preds = ordered_stem_norm(raw_akp_preds)
        # loop through and check if kp in gt and prepare mask accordingly
        dhead_gt, dhead_kp = [], []
        for pred in raw_akp_preds:
            dhead_kp.append(pred)
            if pred in stem_norm(gold_preds):
                dhead_gt.append(1)
            else:
                dhead_gt.append(0)
        dhead_kps.append(dhead_kp)
        dhead_gts.append(dhead_gt)

        curr_hits_m = len(stem_norm(topm_akp_preds) & stem_norm(gold_preds))
        curr_hits_k = len(stem_norm(topk_akp_preds) & stem_norm(gold_preds))
        curr_hits_o = len(stem_norm(topo_akp_preds) & stem_norm(gold_preds))

        # calculate macro scores
        p = curr_hits_m / (len(topm_akp_preds)+eps)
        r = curr_hits_m / (len(gold_preds)+eps)
        # f = (2*p*r)/(p+r+eps)

        p_k = curr_hits_k / (len(topk_akp_preds)+eps)
        r_k = curr_hits_k / (len(gold_preds)+eps)
        # f_k = (2*p_k*r_k)/(p_k+r_k+eps)

        p_o = curr_hits_o / (len(topo_akp_preds)+eps)
        r_o = curr_hits_o / (len(gold_preds)+eps)

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

        topk_pred_kps += len(topk_akp_preds)
        topo_pred_kps += len(topo_akp_preds)
        topm_pred_kps += len(topm_akp_preds)
        total_gt_kps += len(gold_preds)
        # print('\ngt: ', gold_preds)
        # print('preds: ', topm_akp_preds)

        doc_text = '\n' + \
            '\n'.join(wrapper.wrap(' '.join(doc_pt).strip().replace(' ##', '')))
        gt_text = 'Ground Truth: ' + ' ; '.join(list(gold_preds))
        pred_text = 'Pred Text: ' + ' ; '.join(list(topm_akp_preds))
        model_preds.extend([doc_text, gt_text, pred_text])
        # model_preds.extend([' '.join(doc_pt).replace(' ##',''), f'Ground Truth: {list(gold_preds)}', f'Preds: {list(topm_akp_preds)}', '\n'])

    # final metrics on all documents
    p = hitsm / (topm_pred_kps+eps)
    r = hitsm / (total_gt_kps+eps)
    f = (2*p*r)/(p+r+eps)
    # print(f'p,r,f @ M: {p}, {r}, {f}')

    p_k = hitsk / (topk_pred_kps+eps)
    r_k = hitsk / (total_gt_kps+eps)
    f_k = (2*p_k*r_k)/(p_k+r_k+eps)

    p_o = hitso / (topo_pred_kps+eps)
    r_o = hitso / (total_gt_kps+eps)
    f_o = (2*p_o*r_o)/(p_o+r_o+eps)
    # print(f'p,r,f @ K: {p_k}, {r_k}, {f_k}')

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

    # macro score computation
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

    return (res_micro, res_macro), model_preds, (dhead_kps, dhead_gts), (macro_fm_list, macro_fk_list, macro_fo_list)


def display_result_absent(
    doc_text,
    doc_gt,
    doc_pred
):
    # get a random index for display in the batch
    disp_ind = random.randrange(0, len(doc_text))

    doc_text = doc_text[disp_ind][0]
    # no tokens are removed as special tokens are removed by default
    doc_pred = doc_pred[disp_ind]
    doc_gt = doc_gt[disp_ind]

    # list of predictions
    pred_kps = set([x.strip() for x in doc_pred.split(';') if x])
    # list of ground truth
    gt_kps = set([x.strip() for x in doc_gt])
    # restore article (doc_text)
    rest_doc = doc_text.replace(' ##', '')
    return rest_doc, gt_kps, pred_kps

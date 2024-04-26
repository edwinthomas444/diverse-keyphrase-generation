def get_goldkp(doc_list, label_list):
    # key: Ex KP , value: [start_ind, end_ind] of tokens of all words in that key phrase
    ex_kps = {}
    # get the KP spans
    ex_kp, start = '', -1
    for ind, (doc_label, word) in enumerate(zip(label_list, doc_list)):
        # word = doc_list[ind]
        if doc_label == 'B':
            if ex_kp != '': # consecutive B tokens, then register previous one
                ex_kps[ex_kp] = ex_kps.setdefault(ex_kp, [])+[[start, ind]]
            start = ind
            ex_kp = word
        elif doc_label in ['X', 'I'] and start != -1:  # sub-word or inside kp
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
    return ex_kps

def get_goldakp(target):
    akp = target.split(" ; ")
    akp_list = []
    for kp in akp:
        akp_list.append(kp.replace(" ##",""))
    return akp_list
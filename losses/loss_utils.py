import torch
from scipy.optimize import linear_sum_assignment


def hungarian_realignment(dist,
                          target,
                          #attn_mask,
                          ignore_idx):
    """
    dist:   distribution of shape (bs, k, max_seq, vocab)
    target: target of shape (bs, k, max_seq)
    ignore_idx: the indices to ignore for computing the match weights for maximization
    returns: returns realligned target of shape (bs, k, max_seq) for loss computations
    """
    bs, k, max_seq, vocab = dist.shape
    assert (
        bs, k, max_seq) == target.shape, 'Distribution and Target have different shape'

    # convert dist to prob dist (using log softmax to increase the gap of zero and non-zero scores)
    # minimize the scores in this case and values become negative
    # dist = torch.nn.functional.log_softmax(dist, dim=-1)
    dist = torch.nn.functional.softmax(dist, dim=-1)

    # target in y axis and dist in x axis of the graph matrix
    # this is done so that the target can be realigned
    # based on the matrix sum
    dist_repeat = dist.repeat_interleave(k, dim=1)
    target_repeat = target.repeat(1, k, 1)
    out = torch.gather(dist_repeat, 3, target_repeat.unsqueeze(-1)
                       ).squeeze(-1).view(-1, k, k, max_seq)

    # create a mask with 1s over position of PAD tokens (id==0)
    # print('\nign idx: ',ignore_idx)
    mask = target.new_zeros(target.size()).bool()
    for ign_idx in ignore_idx:
        mask |= (target == ign_idx)
    mask = mask.unsqueeze(1)

    # other than masked positions the other positions should be scaled up
    # broad_mask = (~mask).expand(-1,k,k,max_seq)
    # out[broad_mask]*=1e10

    out = out.masked_fill(mask, 0)
    scores = torch.sum(out, dim=3)

    # reorder the target based on linear assignment scores
    # reordered_target = []

    col_inds = []
    for i in range(bs):
        scores_b = scores[i]  # kxk
        _, col_ind = linear_sum_assignment(scores_b, maximize=True)
        col_inds.append(col_ind)
        # reordered_target.append(target[i][col_ind])

    # reordered_target = torch.stack(reordered_target, dim=0)
    # return reordered_target

    return col_inds

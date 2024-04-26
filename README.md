# Launch Distributed Training

```python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --config configs/train.json --ddp```

# Testing
```python test.py --config configs/test.json```

# TODO
1. Add config README


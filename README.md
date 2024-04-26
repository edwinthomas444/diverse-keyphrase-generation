#  Improving Absent Keyphrase Generation with Diversity Heads

Official code for the paper:  
Improving Absent Keyphrase Generation with Diversity Heads  
Edwin Thomas and Sowmya Vajjala  
NAACL 2024  

## Training
```python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --config configs/train.json --ddp```

## Testing
```python test.py --config configs/test.json```


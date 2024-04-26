#  Improving Absent Keyphrase Generation with Diversity Heads

Official code for the paper:  
Improving Absent Keyphrase Generation with Diversity Heads  
Edwin Thomas and Sowmya Vajjala  
NAACL 2024  

## Dataset Preparation
Please refer to [UniKP Official Repo](https://github.com/thinkwee/UniKeyphrase) for dataset preparation details.


## Training
The training pipeline can be run using PyTorch DDP and a config file with the model training configurations. Please refer to files under the `configs` directory to learn about different train settings. PKP and AKG models are trained separately using corresponding `model_map_key` values. 

```python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train.py --config configs/train.json --ddp```

## Testing

After setting the checkpoint path correctly in the test configs, the model can be inferenced. PKP and AKG are tested separately by changing the type of the `model_map_key`.

```python test.py --config configs/test.json```

## Guidelines

1. PKP models can be trained and tested by setting `model_map_key` to `pkp_base`.
2. AKG models can be trained and tested by setting `model_map_key` to `akp_base`.
3. For a full list of supported `model_map_key` types refer to `models/model_maps.py`.

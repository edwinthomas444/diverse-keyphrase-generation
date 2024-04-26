from collators.base_collators import pke_base_collator, akp_base_collator, akp_seq2set_collator, akp_emb2set_collator
from dataset.pipelines import PKPBasePipeline, AKPSeq2SeqPipeline, AKPSeq2SetPipeline, AKPEmb2SetPipeline
from trainers.base_trainers import pke_base_trainer, pke_base_validater, \
    akp_base_trainer, akp_base_validater, \
    akp_seq2set_trainer, akp_seq2set_validater, akp_emb2set_trainer, akp_emb2set_validater

from models.model_loaders import pkp_base_loader, akp_base_loader, akp_seq2set_loader, akp_emb2set_loader
from tokenization.tokenization import WhitespaceTokenizer, UnitTokenizer

maps = {
    'pkp_base': {
        'collator': pke_base_collator,
        'data_pipeline': PKPBasePipeline,
        'trainer': pke_base_trainer,
        'validator': pke_base_validater,
        'model_loader': pkp_base_loader,
        'data_tokenizer': WhitespaceTokenizer,
        'kp_tokenizer': None
    },
    'akp_seq2seq': {
        'collator': akp_base_collator,
        'data_pipeline': AKPSeq2SeqPipeline,
        'trainer': akp_base_trainer,
        'validator': akp_base_validater,
        'model_loader': akp_base_loader,
        'data_tokenizer': WhitespaceTokenizer,
        'kp_tokenizer': None
    },
    'akp_seq2set': {
        'collator': akp_seq2set_collator,
        'data_pipeline': AKPSeq2SetPipeline,
        'trainer': akp_seq2set_trainer,
        'validator': akp_seq2set_validater,
        'model_loader': akp_seq2set_loader,
        'data_tokenizer': WhitespaceTokenizer,
        'kp_tokenizer': UnitTokenizer
    },
    'akp_emb2set': {
        'collator': akp_emb2set_collator,
        'data_pipeline': AKPEmb2SetPipeline,
        'trainer': akp_emb2set_trainer,
        'validator': akp_emb2set_validater,
        'model_loader': akp_emb2set_loader,
        'data_tokenizer': WhitespaceTokenizer,
        'kp_tokenizer': UnitTokenizer
    }
}

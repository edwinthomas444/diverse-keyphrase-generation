from transformers import BertTokenizer
from tokenization.tokenization import WhitespaceTokenizer
from torch.utils.data import Dataset
import sys
sys.path.append('./')


class KPEDataset(Dataset):
    def __init__(self,
                 document_file,
                 label_file,
                 target_file,
                 preprocess_pipeline,
                 skip_none,
                 is_train):

        super().__init__()
        # innitialize pipeline
        self.pipeline = preprocess_pipeline
        self.is_train = is_train

        self.data_points = []
        with open(document_file, 'r', encoding='utf-8') as fd, \
                open(label_file, 'r', encoding='utf-8') as fl, \
        open(target_file, 'r', encoding='utf-8') as ft:
            # store data in-memory: store the raw doc, labels and target lines in lists
            dlines, llines, tlines = fd.readlines(), fl.readlines(), ft.readlines()

            count_null_ext = 0
            for ind, (doc, lab, tgt) in enumerate(zip(dlines, llines, tlines)):
                if len(set(lab.strip().split(' '))) == 1:
                    count_null_ext += 1
                    # continue

                self.data_points.append({
                    'doc': doc,
                    'lab': lab,
                    'tgt': tgt
                })
                # if ind == 100:
                #     break

                
            print('\n Empty extractive text lines: ',
                  count_null_ext, ' out of ', len(dlines))

            # skip none lines
            if skip_none:
                filtered_data_points = [
                    x for x in self.data_points if x['tgt'].strip() != 'none']
                # filtered_data_points = [x for x in self.data_points if x['tgt'].strip(
                # ) != 'none' and len(set(x['lab'].strip().split(" "))) > 1]
                skipped_lines = len(self.data_points)-len(filtered_data_points)
                print('Skipped None Lines: ', skipped_lines)
                self.data_points = filtered_data_points

    # length of whole dataset
    def __len__(self):
        return len(self.data_points)

    # for map style dataset (where DataLoader uses Sampler to get batch indices that call __getitem__(idx)
    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        # pass data points through the preprocess pipeline
        res = self.pipeline(data_point, self.is_train)
        return res


def main():
    from pipelines import PKPBasePipeline, AKPBasePipeline

    # data file paths
    project_path = '/home/thomased/work/codebase'
    doc_path = f'{project_path}/coopsummer2023/UniKP/UniKeyphrase/processed/kp20k.test.seq.in'
    doc_label_path = f'{project_path}/coopsummer2023/UniKP/UniKeyphrase/processed/kp20k.test.seq.out'
    doc_abs_path = f'{project_path}/coopsummer2023/UniKP/UniKeyphrase/processed/kp20k.test.absent'

    # truncate config
    trunc_conf = {
        'len_a': 384
    }

    # create label map
    label_map = {}
    label_list = ['O', 'B', 'I', 'X']
    for ind, lab in enumerate(label_list):
        label_map[lab] = ind

    # innitialize tokenizers
    data_tokenizer = WhitespaceTokenizer()
    model_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # innitialize pipeline (eg: PKP Base Pipeline)
    # pipeline = PKPBasePipeline(
    #     data_tokenizer=data_tokenizer,
    #     model_tokenizer=model_tokenizer,
    #     truncate_config=trunc_conf,
    #     label_map=label_map,
    #     max_len=384
    # )
    pipeline = AKPBasePipeline(
        data_tokenizer=data_tokenizer,
        model_tokenizer=model_tokenizer,
        truncate_config=trunc_conf,
        max_len=384,
        max_len_d=120
    )

    # test the dataset
    dataset = KPEDataset(
        document_file=doc_path,
        label_file=doc_label_path,
        target_file=doc_abs_path,
        preprocess_pipeline=pipeline)

    # print values for any test index
    check_ind = 1
    x = dataset.__getitem__(check_ind)
    # print(len(x['input_ids']))
    # print(len(x['token_type_ids']))
    # print(len(x['attention_mask']))
    # print(len(x['label_ids']))
    # print(x)

    print(len(x['input_ids']))
    print(len(x['attention_mask']))
    print(len(x['decoder_input_ids']))
    print(len(x['decoder_attention_mask']),
          len(x['decoder_attention_mask'][0]))
    print(len(x['labels']))
    print(len(x['doc']))
    print(len(x['gold_akp']))
    print(x)


if __name__ == '__main__':
    main()

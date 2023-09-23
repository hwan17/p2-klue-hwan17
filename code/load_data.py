import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type, test_size, entity):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 's1':dataset[3], 'e1':dataset[4],'entity_02':dataset[5], 's2':dataset[6], 'e2':dataset[7], 'label':label})
    
    if entity:
        df = add_entity_token(out_dataset)
    else:
        df = out_dataset
    
    if test_size == 0 or i == 'blind':
        return df
    elif i != 'blind':
        drop_df = df.drop(df[df['label'] == 40].index)

        train_x, test_x, train_y, test_y = train_test_split(drop_df, drop_df['label'], shuffle=True, stratify=drop_df['label'], test_size=test_size)

        train_x.reset_index(drop=True, inplace = True)
        train_y.reset_index(drop=True, inplace = True)
        test_x.reset_index(drop=True, inplace = True)
        test_y.reset_index(drop=True, inplace = True)

        return train_x, test_x

# tsv 파일을 불러옵니다.
def load_data(dataset_dir, test_size=0.2, entity=True):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type, test_size, entity)
  
  return dataset

def add_entity_token(df):
    dataset = df.copy()
    for i in range(len(df)):
    #     ent1 = df.iloc[i]['sentence'][:df.iloc[i]['s1']] + '[ENT]' + df.iloc[i]['entity_01'] + '[\ENT]' + df.iloc[i]['sentence'][df.iloc[i]['e1']+1:]
        ent1 = '[ENT1]' + dataset.iloc[i]['entity_01'] + '[END1]'
        ent2 = '[ENT2]' + dataset.iloc[i]['entity_02'] + '[END2]'
        if dataset.iloc[i]['s1'] < dataset.iloc[i]['s2']:
            dataset['sentence'].iloc[i] = dataset.iloc[i]['sentence'][:dataset.iloc[i]['s1']] + ent1 + dataset.iloc[i]['sentence'][dataset.iloc[i]['e1']+1:dataset.iloc[i]['s2']] + ent2 + dataset.iloc[i]['sentence'][dataset.iloc[i]['e2']+1:]
        else:
            dataset['sentence'].iloc[i] = dataset.iloc[i]['sentence'][:dataset.iloc[i]['s2']] + ent1 + dataset.iloc[i]['sentence'][dataset.iloc[i]['e2']+1:dataset.iloc[i]['s1']] + ent2 + dataset.iloc[i]['sentence'][dataset.iloc[i]['e1']+1:]
    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
  sep = tokenizer.special_tokens_map["sep_token"]
  concat_entity = list(dataset['entity_01'] + sep + dataset['entity_02'])
#   concat_entity = []
#   for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
#     temp = ''
#     temp = e01 + '[SEP]' + e02
#     concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  return tokenized_sentences

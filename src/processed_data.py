import config
from transformers import AutoTokenizer
from datasets import load_dataset


def process():
    print("开始处理数据")
    
    tokenizer=AutoTokenizer.from_pretrained(config.PRETRAIN_MODEL_PATH)
    dataset=load_dataset('json',data_files=str(config.RAW_DATA / 'data.jsonl'))['train']
    
    lable_list=[]
    for labels in dataset['labels']:
        lable_list.extend(labels)
    lable_list=list(set(lable_list))
    #print("标签列表:",lable_list)
    
    with open(config.PROCESSED_DATA / 'label_list.txt','w',encoding='utf-8') as f:
        for label in lable_list:
            f.write(label+'\n')
            
    #print(dataset[0])
    
    #将字符label映射为数字label
    # dataset=dataset.cast_column('labels',ClassLabel(names=lable_list))
    #错误！因为labels是list[str],不是str,不能直接cast,只能在map中处理
    
    def encode_batch(example):
        inputs=tokenizer(
            example['text'],
            truncation=True,
            is_split_into_words=True#将list[str]视为一个序列，而不是多个序列
        )
        
        #将标签对其，因为一个string可能被拆分为多个token
        all_labels=[]
        for i,labels in enumerate(example['labels']):#遍历一个batch中的每个样本获得labels
            word_ids=inputs.word_ids(batch_index=i)#获取第i个样本的word_ids映射,padding部分为None,非padding部分为对应的word索引
            label_ids=[]
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)#padding部分设置为-100,因为pytorch的crossentropyloss会忽略-100
                else:
                    label=labels[word_idx]#获取对应word的label
                    label_id=lable_list.index(label)#将label转换为数字id
                    label_ids.append(label_id)
            all_labels.append(label_ids)
        inputs['labels']=all_labels
        return inputs
    
    dataset=dataset.map(
        encode_batch,
        batched=True,
        remove_columns=['text','labels']#移除原始列
    )
    
    # print(len(dataset[0]['labels']))
    # print(len(dataset[0]['input_ids']))
    
    # print(tokenizer.decode(dataset[0]['input_ids']))
    # labelindex=dataset[0]['labels'][1:-1]
    # print(labelindex)
    # labels=[lable_list[index] for index in labelindex]
    # print(labels)
    # print(dataset[0])
    
    #划分数据集并保存
    dataset_dict=dataset.train_test_split(test_size=0.2,seed=42)
    dataset_dict.save_to_disk(str(config.PROCESSED_DATA))
    
    print("数据处理完成")


if __name__ == '__main__':
    process()
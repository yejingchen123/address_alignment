import config
from transformers import AutoTokenizer,DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from datasets import load_from_disk

def get_dataloder(istrain=True):
    data_path=""
    if istrain:
        data_path=str(config.PROCESSED_DATA / 'train')
        shuffle=True
    else:
        data_path=str(config.PROCESSED_DATA / 'test')
        shuffle=False
    dataset=load_from_disk(data_path)
    dataset.set_format(type='torch')
    tokenizer=AutoTokenizer.from_pretrained(str(config.PRETRAIN_MODEL_PATH))
    #print(dataset[0])
    collate_fn=DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        #label_pad_token_id=-100,#默认
        padding=True
    )
    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=shuffle,collate_fn=collate_fn)




if __name__ == "__main__":
    dataloder=get_dataloder(istrain=True)
    for batch in dataloder:
        for k,v in batch.items():
            print(f'{k}: {v}')
        break
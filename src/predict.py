import config
from transformers import AutoTokenizer,AutoModelForTokenClassification
import torch
import numpy as np

def process_text(text_list,tokenizer):
    all_text=[]
    for text in text_list:
        text=text.strip()
        text=text.replace('，','')
        text=text.replace('。','')
        text=text.replace(' ','')
        all_text.append(list(text))
    
    intputs=tokenizer(all_text,truncation=True,padding=True,is_split_into_words=True,return_tensors='pt')
    return intputs

def process_preds(preds,id2label,tokenizer,input_ids):
    map={
        'city':'城市', 
        'phone':'电话', 
        'O':'其他', 
        'district':'街区', 
        'name':'姓名',
        'prov':'省份',
        'town':'县镇',
        'detail':'详细地址'
    }
    results=[]
    for pred,input_id in zip(preds,input_ids):
        result={}
        for index,label_index in enumerate(pred):
            label=id2label[label_index]
            label=label.split('-')[1] if len(label)>1 else label
            if label=='O':
                continue
            if map[label] not in result:
                result[map[label]]=""
            result[map[label]]+=tokenizer.decode(input_id[index])
        
        results.append(result)
    return results

def predict_batch(model,inputs,device,tokenizer):
    model.eval()
    with torch.no_grad():
        inputs={k:v.to(device) for k,v in inputs.items()}
        outputs=model(**inputs)
        preds=torch.argmax(outputs.logits,dim=-1).cpu().numpy().tolist()
        input_ids=inputs['input_ids'].cpu().numpy().tolist()
    results=process_preds(preds,model.config.id2label,tokenizer,input_ids)
    return results
            
            
def predict(text):
    
    #设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #tokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(config.PRETRAIN_MODEL_PATH))
    
    #模型
    model=AutoModelForTokenClassification.from_pretrained(str(config.CHECKPOINT_BEST_DIR)).to(device)
    
    inputs=process_text(text,tokenizer)
    results=predict_batch(model,inputs,device,tokenizer)
    
    return results
    
    # for result in results:
    #     for key,value in result.items():
    #         print(f'{key}:{value}')
    #     print("="*20)
    


if __name__ == "__main__":
    text=['小明18888888888江苏省南京市江宁区兰台公寓','小王18888888888四川省成都市犀浦镇犀安路999号']
    results=predict(text)
    for result in results:
        for key,value in result.items():
            print(f'{key}:{value}')
        print("="*20)

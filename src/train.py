import config
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer,AutoModelForTokenClassification
from dataset import get_dataloder
from tqdm import tqdm
import time


def train_one_step(model,inputs,optimizer,device):
    model.train()
    inputs={k:v.to(device) for k,v in inputs.items()}
    outputs=model(**inputs)
    l=outputs.loss
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    return l.item()

def train():
    #设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE: {device}')
    
    #数据
    dataloader=get_dataloder(istrain=True)
    
    #label_list
    label_list=[]
    with open(config.PROCESSED_DATA / 'label_list.txt','r',encoding='utf-8') as f:
        for line in f:
            label_list.append(line.strip())
    # print(f'NUM_LABELS: {num_labels}')
    label2id={label:index for index,label in enumerate(label_list)}
    id2label={index:label for index,label in enumerate(label_list)}
    
    #模型
    model=AutoModelForTokenClassification.from_pretrained(
        str(config.PRETRAIN_MODEL_PATH),
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    
    #优化器
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.LEARNING_RATE)
    
    #日志
    writer=SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d-%H-%M-%S'))
    
    best_loss=float('inf')
    
    step=1
    
    for epoch in range(1,config.EPOCH+1):
        for batch in tqdm(dataloader,desc=f'Training: Epoch {epoch}'):
            loss=train_one_step(model,batch,optimizer,device)
            
            if step%config.SAVE_STEPS==0:
                tqdm.write(f'step={step},loss:{loss:.4f}')
                writer.add_scalar('loss',loss,step)
                
                if loss<best_loss:
                    best_loss=loss
                    model.save_pretrained(config.CHECKPOINT_BEST_DIR)
                    tqdm.write(f'Best model saved.')
                
                model.save_pretrained(config.CHECKPOINT_LAST_DIR)
                tqdm.write(f'Last model saved.')
                
            step+=1
    



if __name__ == "__main__":
    train()

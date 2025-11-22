import config
import torch
from transformers import AutoModelForTokenClassification
from sklearn.metrics import accuracy_score,f1_score
from dataset import get_dataloder
from tqdm import tqdm
import numpy as np


def compute_metrics(preds,labels):
    return {
        'accuracy':accuracy_score(labels,preds),
        'f1':f1_score(labels,preds,average='weighted')
    }

def update(all_preds,all_labels,preds_batch,labels_batch,ignore_index=-100):
    
    #将preds_batch和labels_batch的形状转换为一维
    preds_flat = np.array(preds_batch).flatten()
    labels_flat = np.array(labels_batch).flatten()
    #忽略-100的预测结果
    mask = labels_flat != ignore_index
    preds_flat = preds_flat[mask]
    labels_flat = labels_flat[mask]
    #更新预测结果和标签
    all_preds=np.concatenate((all_preds,preds_flat))
    all_labels=np.concatenate((all_labels,labels_flat))
    
    return all_preds,all_labels
    
    
def evaluate_model(model,dataloader,device):
    model.eval()
    all_preds=np.array([])
    all_labels=np.array([])
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='evaluating'):
            batch={k:v.to(device) for k,v in batch.items()}
            
            outputs=model(**batch)#outputs.logits:[batch_size,seq_len,num_labels]
            preds_batch=torch.argmax(outputs.logits,dim=-1).cpu().numpy().tolist()
            labels_batch=batch['labels'].cpu().numpy().tolist()
            all_preds,all_labels=update(all_preds,all_labels,preds_batch,labels_batch)
            

    return all_preds,all_labels
            
            
            

def evaluate():
    
    #设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #数据
    dataloader=get_dataloder(istrain=False)
    
    #模型
    model=AutoModelForTokenClassification.from_pretrained(str(config.CHECKPOINT_BEST_DIR)).to(device)
    
    all_preds,all_labels=evaluate_model(model,dataloader,device)
    
    metrics=compute_metrics(all_preds,all_labels)
    print(f'accuracy: {metrics["accuracy"]:.3f},f1: {metrics["f1"]:.3f}')


if __name__ == '__main__':
    evaluate()

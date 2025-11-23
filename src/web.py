from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from predict import predict

class Text(BaseModel):
    text: list[str]
    
class Prediction(BaseModel):
    predict:list[dict[str,str]]
    
app = FastAPI()

@app.post("/predict")
def predict_text(text:Text):
    prediction=predict(text.text)
    return Prediction(predict=prediction)

def web_run():
    uvicorn.run(app, host="0.0.0.0", port=8000)
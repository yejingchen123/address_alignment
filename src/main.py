from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(usage="python main.py --mode [train/evaluate/predict/web]")
    parser.add_argument("--mode", type=str, required=True, help="mode:train/evaluate/predict/web")
    agrs = parser.parse_args()
    
    if agrs.mode=="train":
        from train import train
        train()
    elif agrs.mode=="evaluate":
        from evaluate import evaluate
        evaluate()
    elif agrs.mode=="predict":
        from predict import predict
        predict()
    elif agrs.mode=="web":
        from web import web_run
        web_run()
    else:
        print("Invalid mode,Plase choosee from [train/evaluate/predict/web]")
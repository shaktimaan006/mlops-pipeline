import logging
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import json

# make logger file 
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# config logger
logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# make handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_log_path=os.path.join(log_dir,'model_evaluation.log')
file_handler=logging.FileHandler(file_log_path)
file_handler.setLevel('DEBUG')

# formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# now do the model evaluation
"""
1. load model
2. load test data
3. evaluate model with test data
4. save the evaluation metrics
"""

def load_model(model_path:str):
    """This funtion loades the pkl model """
    try:
        with open(model_path,'rb') as file:
            model=pickle.load(file)
        logger.debug('Model loaded from %s',model_path)
        return model
    except FileNotFoundError:
        print('s')
        logger.error('File not found at %s',model_path)
    except Exception as e:
        print('h')
        logger.error("Somthing went wrong while loading model")


def load_data(data_path:str)->pd.DataFrame:
    """load test data from path to evaluate model"""
    try:
        df=pd.read_csv(data_path)
        logger.debug('Data loaded sucessfully from %s ',data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(x_test:np.ndarray,y_test:np.ndarray,model)->dict:
    """Evaluate model """
    try:
        y_pred=model.predict(x_test)
        y_pred_prob=model.predict_proba(x_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_prob)

        metrics={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
                 }
        logger.debug('model evaluation is completed')
        return metrics
    except Exception as e:
        logger.error('Soming went wrong while evalution of model :%s',e)

# 5. save evaluation metrics
def save_metrics(metrics,path:str):
    """Save metrics of model evaluation"""
    try:
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug("Metrics saved in :%s",path)
    except Exception as e:
        logger.error("Somthing went wrong while saving metrics")


def main():
    try:
        print('this is main')
        model=load_model('./models/model.pkl')
        df=load_data('./data/processed/test_tfidf.csv')

        x_test=df.iloc[:,:-1]
        y_test=df.iloc[:,-1]

        metrics=evaluate_model(x_test,y_test,model)
        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error("Something went wrong in model evaluation file")
        print(e)

if __name__=='__main__':
    main()


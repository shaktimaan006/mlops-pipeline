import logging
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

level='DEBUG'
# make log file
log_path='logs'
os.makedirs(log_path,exist_ok=True)

# configure logger
logger=logging.getLogger('model_building')
logger.setLevel(level)

# create console handler
console_handler=logging.StreamHandler()
console_handler.setLevel(level)

# create file handler
file_log_dir=os.path.join(log_path,'model_building.log')
file_handler=logging.FileHandler(file_log_dir)
file_handler.setLevel(level)

# create formatter
formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# load data
def load_data(file_path:str)->pd.DataFrame:
    """Load data from cs"""
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse')
        raise
    except FileNotFoundError as e:
        logger.error('File not fount ')
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data')
        raise


# train model
def train_model(x_train,y_train,params)->RandomForestClassifier:
    """Train random forest model 
    on x_train,y_train
    """
    try:
        if(x_train.shape[0]!=y_train.shape[0]):
            raise ValueError("The no of samples in x_train and y_train must be the same")

        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.debug('Model training stated')
        clf.fit(x_train,y_train)
        logger.debug('Model training completed')

        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s',e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s',e)


# save model
def save_model(model,file_path:str)->None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("Model saved as pickle")
    except FileNotFoundError as e:
        logger.error("File not found at %s ",file_path)
        raise
    except Exception as e:
        logger.error("Somthing went wrong in saving model")
        raise

    
# main file
def main():
    try:
        params={'n_estimators':25,'random_state':2}
        train_data=load_data('./data/processed/train_tfidf.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
        clf=train_model(x_train,y_train,params)

        model_path='./models/model.pkl'
        save_model(clf,model_path)
    except Exception as e:
        logger.error('Failed to complete model fetching')
        raise



if __name__=='__main__':
    main()
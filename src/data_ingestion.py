import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import yaml

# make a log directory
log_dir='logs'
os.makedirs(log_dir,exist_ok=True) # exist_ok:- not give error if already folder exist

# maker a logger object
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# make console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# make a file handler
log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# make a formatter(format of logs)
formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_data(data_url:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data sucessfully loded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error("something went wrong while loding data %s",e)
        raise



def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """Preprocess the data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug("Data preprocessing completed (only colomn droped and remaining renamed to text,target)")
        return df
    except KeyError as e:
        logger.error("Missing column in Dataframe: %s",e)
        raise
    except Exception as e:
        logger.error("Something went wrong while preprocessing the data: %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """Save the train and test dataset"""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and test data saved to %s',raw_data_path)
    except Exception as e:
        logger.error("Unexpected error while saving the data %s",e)
        raise

def main():
    try:
        test_size=0.2
        data_path="https://raw.githubusercontent.com/shaktimaan006/Data/refs/heads/main/spam.csv"
        df=load_data(data_path)
        new_df=preprocess_data(df)
        train_data,test_data=train_test_split(new_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error("something went wrong while saving the data %s",e)



if __name__=='__main__':
    main()
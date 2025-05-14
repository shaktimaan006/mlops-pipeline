import os
import logging
import nltk
import string
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

# define column
input_column='text'
target_column='target'

# make a log directory
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# create a logger obj and do configuration
logger=logging.getLogger('data_preprocess')
logger.setLevel('DEBUG')

# create handler

# create stream handler
stream_handler=logging.StreamHandler()
stream_handler.setLevel('DEBUG')

# create file handler
logfile_path=os.path.join(log_dir,'data_preprocess.log')
file_handler=logging.FileHandler(logfile_path)
file_handler.setLevel('DEBUG')

# create formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add handler to logger obj
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# the data pre processing part
def transform_text(text):
    """Take text column as input and apply lowering,tokenize,remove stopword and punctuation,steaming"""
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)


# Preprocess on train and test data (column=text,target)
def preprocess(df:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info("starting preprocessing on data....")
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        # remove duplicate rows
        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicate removed')

        # apply text transformation
        df.loc[:,input_column]=df[input_column].apply(transform_text)
        logger.debug('Text column transfered')
        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise



def main():
    """main fn to load raw data,process it and save the processed data"""
    try:
        # fetch data
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')

        logger.debug('Fetched complete')
        # transform data
        train_processed_data = preprocess(train_data)
        test_processed_data = preprocess(test_data)


        logger.debug('Text preprocessing completed')

        # store data inside data/processed
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__=='__main__':
    main()


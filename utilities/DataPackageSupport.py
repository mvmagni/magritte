from yellowbrick.target import ClassBalance
import pandas as pd
from tqdm import tqdm
import uuid
from cleantext import clean
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import contractions
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler


def create_unique_column(dataFrame,
                         unique_column='uuid'):
    # Needs to be rewritten to get rid of "making changes to a slice of the data" warning
    dataFrame['uuid'] = dataFrame.apply(lambda _: uuid.uuid4(), axis=1)


def clean_text_column(dataFrame,
                      data_column,
                      data_package_params,
                      process_count
                      ):
    #tqdm.pandas()
    #tDf = dataFrame.copy()
    #print(f'P{process_count} ', end='')
    dataFrame[data_column] = dataFrame[data_column].apply(lambda x: processText(inputText=str(x),
                                                                             data_package_params=data_package_params))
    print(f'P{process_count} ', end='')
    return dataFrame

def processText(inputText, data_package_params):
    x = inputText

    if data_package_params.lower:
        x = x.lower()

    if data_package_params.fix_contractions:
        x = fix_contractions(document=x)
    
    x = clean(x,
                lower=False,  # lowercase text set to False. Done outside of this package
                fix_unicode=data_package_params.fix_unicode,  # fix various unicode errors
                to_ascii=data_package_params.to_ascii,  # transliterate to closest ASCII representation
                no_line_breaks=data_package_params.no_line_breaks,
                # fully strip line breaks as opposed to only normalizing them
                no_urls=data_package_params.no_urls,  # replace all URLs with a special token
                no_emails=data_package_params.no_emails,  # replace all email addresses with a special token
                no_phone_numbers=data_package_params.no_phone_numbers,  # replace all phone numbers with a special token
                no_numbers=data_package_params.no_numbers,  # replace all numbers with a special token
                no_digits=data_package_params.no_digits,  # replace all digits with a special token
                no_currency_symbols=data_package_params.no_currency_symbols,
                # replace all currency symbols with a special token
                no_punct=data_package_params.no_punct,  # remove punctuations
                replace_with_punct=data_package_params.replace_with_punct,
                # instead of removing punctuations you may replace them
                replace_with_url=data_package_params.replace_with_url,
                replace_with_email=data_package_params.replace_with_email,
                replace_with_phone_number=data_package_params.replace_with_phone_number,
                replace_with_number=data_package_params.replace_with_number,
                replace_with_digit=data_package_params.replace_with_digit,
                replace_with_currency_symbol=data_package_params.replace_with_currency_symbol,
                lang=data_package_params.lang  # set to 'de' for German special handling
                )

    if data_package_params.lemmatize:
        x = lemmatize(document=x)

    if data_package_params.remove_small_tokens:
        x = remove_small_tokens(document=x,
                            data_package_params=data_package_params)

    if data_package_params.remove_stopwords:
        x = remove_stopwords(document=x,
                                data_package_params=data_package_params)    

    

    return x

#Removing certain sized words
def remove_small_tokens(document,
                        data_package_params):
    nltk.download('punkt', quiet=True)
    new_doc = ' '.join([i for i in document.split() if len(i) > data_package_params.min_token_size])
    return new_doc


def lemmatize(document):
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    stemmer = WordNetLemmatizer()    
    #new_doc = ' '.join([stemmer.lemmatize(word) for word in word_tokenize(document)])
    new_doc = ' '.join([stemmer.lemmatize(word) for word in document.split()])
    return new_doc

def fix_contractions(document):
    return contractions.fix(document)

def remove_stopwords(document,
                     data_package_params):
    
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop = stopwords.words(data_package_params.stopword_language)
    if data_package_params.custom_stopwords is not None:
        stop.extend(data_package_params.custom_stopwords)
    
    #new_doc = ' '.join([word for word in word_tokenize(document) if word not in stop])
    new_doc = ' '.join([word for word in document.split() if word not in stop])
    return new_doc


def process_TFIDF(dataFrame,
                  data_column,
                  columns_in_output,
                  max_features=100):
    # print(f'process_TFIDF: original dataFrame shape = {dataFrame.shape}')
    # print(f'process_TFIDF: columns in output: {columns_in_output}')
    v = TfidfVectorizer(max_features=max_features)
    x = v.fit_transform(dataFrame[data_column])
    feature_names = v.get_feature_names_out()
    tDf = pd.DataFrame(data=x.toarray(), columns=feature_names)

    # print(f'process_TFIDF: tDf shape after vectorize = {tDf.shape}')

    if columns_in_output is None:
        # No extra columns requested. return as is
        return tDf
    else:
        # Get a copy of the extra columns (e.g. uuid, target column)
        tDfUn = dataFrame[columns_in_output].copy()
        # print(f'process_TFIDF: df extra columns shape to be merged = {tDfUn.shape}')

        # reset indexes on both otherwise pd.concat goes funky
        tDfUn.reset_index(inplace=True, drop=True)
        tDf.reset_index(inplace=True, drop=True)

        # Merge extra columns to encoded frame
        retDF = pd.concat([tDfUn, tDf], axis=1)
        # print(f'process_TFIDF: returning frame with shape = {retDF.shape}')
        # Return encoded frame with extra columns
        return retDF


def trainTestSplit(dataFrame,
                   train_size=0.8,
                   random_state=765,
                   stratifyColumn=None,
                   shuffle=True):
    origDataSize = len(dataFrame)
    indent = '---> '
    if stratifyColumn is None:
        train, test = train_test_split(dataFrame,
                                       train_size=train_size,
                                       random_state=random_state,
                                       shuffle=shuffle
                                       )
    else:
        print(f'Calling train_test_split with shuffle: {shuffle}')
        train, test = train_test_split(dataFrame,
                                       train_size=train_size,
                                       random_state=random_state,
                                       stratify=dataFrame[[stratifyColumn]],
                                       shuffle=shuffle
                                       )

    print(f'Completed train/test split (train_size = {train_size}):')
    print(f'{indent}Original data size: {origDataSize}')
    print(f'{indent}Training data size: {len(train)}')
    print(f'{indent}Testing data size: {len(test)}')
    if stratifyColumn is None:
        print(f'{indent}Not stratified on any column')
    else:
        print(f'{indent}Stratified on column: {stratifyColumn}')

    return train, test


def classBalanceUndersample(dataFrame,
                            columnName,
                            sampleSize=None,
                            alreadyBalanced=False):
    # Display the initial state
    tDf = dataFrame.copy()
    displayClassBalance(data=tDf,
                        columnName=columnName)

    if alreadyBalanced:
        print("Classes already balanced")
        return

    # Not balanced, need to get some info to get size to balance to
    ttlColName = 'ttlCol'

    # If no size specified then calculate based on smallest class
    if sampleSize is None:
        # Find the sample size by finding which group/class is smallest
        tDfSize = tDf.groupby([columnName]).size().to_frame(ttlColName).sort_values(by=ttlColName)
        tDfSize.reset_index(inplace=True)
        sample_size = pd.to_numeric(tDfSize[ttlColName][0])
        sample_class = tDfSize[columnName][0]
        print(f'Undersampling data to match min class: {str(sample_class)} of size: {sample_size}')
    else:
        # Sample size given so use that to balance
        sample_size = sampleSize

    # Do the sampling
    tDf = tDf.groupby(columnName, group_keys=False).apply(lambda x: x.sample(sample_size))
    tDf.reset_index(drop=True, inplace=True)

    displayClassBalance(data=tDf,
                        columnName=columnName,
                        verbose=True)

    # Return the balanced dataset
    return tDf


def classBalanceOversample(dataFrame,
                           columnName,
                           random_state=987,
                           verbose=True,
                           show_records=5):
    # Display the initial state
    tDf = dataFrame.copy()
    displayClassBalance(data=tDf,
                        columnName=columnName,
                        verbose=verbose,
                        showRecords=show_records)
    # Split dataset into X/Y
    y_unbal = tDf[[columnName]].copy()
    x_unbal = tDf.copy()
    
    
    print(f'Oversampling data to match max class')
    ros = RandomOverSampler(random_state=random_state)
    x_bal, y_bal = ros.fit_resample(x_unbal, y_unbal)
    
    
    displayClassBalance(data=x_bal,
                        columnName=columnName,
                        verbose=verbose,
                        showRecords=show_records)

    # Return the balanced dataset
    # Note that we left all columns in x_unabl
    # so x_bal will be balanced with all required columns
    # no concat/merge required.
    return x_bal


def displayClassBalance(data,
                        columnName,
                        verbose=False,
                        showRecords=5):
    ttlColName = 'ttlCol'

    visualizer = ClassBalance()
    visualizer.fit(data[columnName])  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

    if verbose:
        tDfSize = data.groupby([columnName]).size().to_frame(ttlColName).sort_values(by=ttlColName).copy()
        tDfSize.reset_index(inplace=True)
        display(tDfSize.head(showRecords))

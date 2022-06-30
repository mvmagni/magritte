from yellowbrick.target import ClassBalance
import pandas as pd
from tqdm import tqdm
import uuid
from cleantext import clean
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


def create_unique_column(dataFrame,
                         unique_column='uuid'):
    dataFrame[unique_column] = [uuid.uuid4() for _ in range(len(dataFrame.index))]


def clean_text_column(dataFrame,
                      data_column,
                      data_package_params=None):
    tqdm.pandas()
    print(f'Cleaning text column...')
    if data_package_params is None:  # Use defaults from cleanText package
        dataFrame[data_column] = dataFrame[data_column].progress_apply(lambda x: clean(x))
    else:  # Use those stored in data_package_params
        dataFrame[data_column] = dataFrame[data_column].progress_apply(lambda x: cleanText(inputText=x,
                                                                                           data_package_params=data_package_params))


def cleanText(inputText, data_package_params):
    x = clean(inputText,
              fix_unicode=data_package_params.fix_unicode,  # fix various unicode errors
              to_ascii=data_package_params.to_ascii,  # transliterate to closest ASCII representation
              lower=data_package_params.lower,  # lowercase text
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

    return x


def remove_stopwords(dataFrame,
                     data_column,
                     stopword_language='english'):
    tqdm.pandas()
    print(f'Removing stopwords...')
    nltk.download('stopwords')
    stop = stopwords.words(stopword_language)
    dataFrame[data_column] = dataFrame[data_column].progress_apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop]))


def process_TFIDF(dataFrame,
                  data_column,
                  columns_in_output,
                  max_features=100):
    v = TfidfVectorizer(max_features=max_features)
    x = v.fit_transform(dataFrame[data_column])
    feature_names = v.get_feature_names_out()
    tDf = pd.DataFrame(data=x.toarray(), columns=feature_names)

    if columns_in_output is None:
        # No extra columns requested. return as is
        return tDf
    else:
        # Get a copy of the extra columns (e.g. uuid, target column)
        tDfUn = dataFrame[columns_in_output].copy()

        # Merge extra columns to encoded frame
        retDF = pd.concat([tDfUn, tDf], axis=1)

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

    # Return the balance dataset
    return tDf


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

import DataPackageSupport as dps
import pickle
import gzip
from dataclasses import dataclass, field
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataPackage:
    __version = 0.2

    def __init__(self,
                 original_data,
                 unique_column,
                 target_column,
                 data_column,
                 data_package_params):
        
        self.isProcessed = False
        self.targetColumn = target_column
        self.dataColumn = data_column
        
        self.isImportantFeaturesApplied = False
        self.isLabelEncoded = False            
        self.labelEncoder = preprocessing.LabelEncoder()
        
        self.__setDataPackageParams(data_package_params=data_package_params)
        self.__setOrigData(original_data, unique_column)

    # Set the original dataFrame (pandas)
    def __setOrigData(self, origData, unique_column):
        self.origData = origData
        self.__setUniqueColumn(unique_column)
        self.isOrigDataLoaded = True

        # Keep orig data untainted for process/integrity
        # WorkingData is what we will use for processing
        self.__setWorkingData(origData=origData)

        # Do a label encoding on target column
        self.__encodeTargetColumn()

        # A new dataframe means we need to reset our work
        self.__resetWork()


    # label encode target column
    def __encodeTargetColumn(self):
        if not self.isLabelEncoded and self.targetColumn is not None:
            print(f'Target column found. Label encoding target column.')
            self.workingData[self.targetColumn] = self.labelEncoder.fit_transform(self.workingData[self.targetColumn])
            self.isLabelEncoded = False
            


    # Set working data. Leaves original dataframe intact
    def __setWorkingData(self,
                         origData):
        self.workingData = origData.copy()
        self.isWorkingDataLoaded = True
        self.setDataFeatures(feature_list = self.workingData.columns)

    # give feature_list including unique and target columns
    def setDataFeatures(self,
                        feature_list):
        # Get and set features listing, removing unique and target columns
        self.dataFeatures = list(feature_list)
        # Remove unique and target column
        self.dataFeatures.remove(self.uniqueColumn)
        self.dataFeatures.remove(self.targetColumn)

    # if the data gets changed then we need to
    # invalidate all the results/work done previously
    def __resetWork(self):
        self.isBalanced = False
        self.isCleaned = False
        self.isEncoded = False
        self.isStopWorded = False

        self.__clearTrainTestData()

    # Unique column required. If none present then create one
    def __setUniqueColumn(self, unique_column):
        DEFAULT_UNIQUE = 'uuid'

        # If no unique column then create one
        if unique_column is None:
            dps.create_unique_column(dataFrame=self.origData,
                                     unique_column=DEFAULT_UNIQUE)
            self.uniqueColumn = DEFAULT_UNIQUE
        else:
            self.uniqueColumn = unique_column

    # Set the dataPackageParams object. Optional functionality
    def __setDataPackageParams(self,
                               data_package_params):
        self.data_package_params = data_package_params
        self.hasDataPackageParams = True

    # Process the datapackage. Called from __setDataPackageParams
    def processDataPackage(self, 
                           num_cores=6,
                           random_state=786,
                           verbose=True,
                           show_records=5):
        # Confirm it hasn't already been processed
        if self.isProcessed:
            print(f'DataPackage has already been processed')
            return

        self.display()
        print(f'Processing data package with provided parameters')


        # General message for not balancing
        if not self.data_package_params.balance_dataset:
            print(f'Not balancing dataset. balance_dataset=False')


        if self.data_package_params.balance_dataset and self.data_package_params.balance_type == 'undersample':
            # classbalance undersample
            self.classBalanceUndersample(sampleSize=self.data_package_params.undersample_size)

        ##################################
        # Text Cleaning
        self.processText(num_cores=num_cores)

        ##################################
        # Text encoding

        # process_TFIDF
        self.process_TFIDF(max_features=self.data_package_params.max_features)


        # splittraintest
        self.splitTrainTest(stratifyColumn=self.data_package_params.stratifyColumn,
                            train_size=self.data_package_params.train_size,
                            random_state=self.data_package_params.random_state,
                            shuffle=self.data_package_params.shuffle)

        
        # Class balance oversample - has to occur after train/test split
        # oversample only on training data
        if self.data_package_params.balance_dataset and self.data_package_params.balance_type == 'oversample':
            
            self.classBalanceOversample(verbose=verbose,
                                        show_records=show_records,
                                        random_state=random_state)
      
        print(f'')
        print(f'Processing data package has been completed')
        print(f'')
        self.isProcessed = True
        self.display()


    # Function for cleaning text, lemmatization, etc
    # Basic functionality included now, to be expanded later
    # Goal is to keep this to one function if possible
    def processText(self, num_cores=4):
        
        # Stores parameters for sending to function to analyze frame
        func_input = []

        df_split = np.array_split(self.workingData, num_cores)
        
        for count, df_part in enumerate(df_split, start=1):
            func_input.append([df_part, self.dataColumn, self.data_package_params, count])

        print(f'Spawning {num_cores} processes to process text')
        print(f'Processes completed: [ ', end='')
        myDF = None
        pool = Pool(num_cores)
        myDF = pd.concat(pool.starmap(dps.clean_text_column, func_input))
        
        pool.close()
        pool.join()
        print(f']')

        self.workingData = myDF.copy()
        self.isCleaned = True
        self.isStopWorded = self.data_package_params.remove_stopwords


    def process_TFIDF(self,
                      max_features=100,
                      columns_in_output=None):
        if self.isEncoded:
            print(f'Working dataset already encoded')
            return

        # No additional columns specified manually.
        # Add in unique column and target column
        print(f'')
        print(f'Encoding to TF-IDF with max_features={max_features}')
        if columns_in_output is None:
            add_cols = [self.uniqueColumn,
                        self.targetColumn]
        else:
            add_cols = columns_in_output

        encodedFrame = dps.process_TFIDF(dataFrame=self.getWorkingData(),
                                         data_column=self.dataColumn,
                                         max_features=max_features,
                                         columns_in_output=add_cols)

        self.__setWorkingData(encodedFrame)
        self.isEncoded = True
        print(f'Encoding completed. Feature list:')
        print(self.dataFeatures)
        print(f'')

    def __setTrainData(self, trainData):
        self.trainData = trainData
        self.isTrainDataLoaded = True

    def getTrainData(self):
        if self.isTrainDataLoaded:
            return self.trainData
        else:
            print('Train data has not been loaded')

    def __setTestData(self, testData):
        self.testData = testData
        self.isTestDataLoaded = True

    def getTestData(self):
        if self.isTestDataLoaded:
            return self.testData
        else:
            print("Test data has not been loaded")

    def __clearOrigData(self):
        self.origData = None
        self.isOrigDataLoaded = False

    def __clearTrainTestData(self):
        self.isTrainTestSplit = False
        self.isTrainDataLoaded = False
        self.trainData = None

        self.isTestDataLoaded = False
        self.testData = None

    def splitTrainTest(self,
                       stratifyColumn=None,
                       train_size=0.8,
                       random_state=765,
                       shuffle=True
                       ):

        if stratifyColumn is None:
            stratifyColumn = self.targetColumn

        train, test = dps.trainTestSplit(dataFrame=self.getWorkingData(),
                                         train_size=train_size,
                                         random_state=random_state,
                                         stratifyColumn=stratifyColumn,
                                         shuffle=shuffle)

        self.__setTrainData(train)
        self.__setTestData(test)
        self.isTrainTestSplit = True

    def getOrigData(self):
        if self.isOrigDataLoaded:
            return self.origData
        else:
            print("Original data frame is not loaded")

    def getWorkingData(self):
        return self.workingData

    def display(self):
        emptySpace = ''
        indent = emptySpace + '---> '

        print(f'{emptySpace}DataPackage summary')
        print(f'{emptySpace}Attributes:')

        print(f'{indent}uniqueColumn: {self.uniqueColumn}')
        print(f'{indent}dataColumn: {self.dataColumn}')
        print(f'{indent}targetColumn: {self.targetColumn}')

        print(f'{emptySpace}Data:')
        print(f'{indent}isOrigDataLoaded: {self.isOrigDataLoaded}')
        print(f'{indent}isWorkingDataLoaded: {self.isWorkingDataLoaded}')
        
        print(f'{indent}isTrainDataLoaded: {self.isTrainDataLoaded}')
        print(f'{indent}isTestDataLoaded: {self.isTrainDataLoaded}')
        print(f'')

        print(f'{emptySpace}Original Data:')
        print(f'{indent}original data shape: {self.origData.shape}')
        
        print(f'{emptySpace}Working Data:')
        print(f'{indent}working data shape: {self.workingData.shape}')
        print(f'')

        print(f'{emptySpace}Process:')
        print(f'{indent}isProcessed: {self.isProcessed}')
        print(f'{indent}isCleaned: {self.isCleaned}')
        print(f'{indent}isStopWorded: {self.isStopWorded}')
        print(f'{indent}isBalanced: {self.isBalanced}')
        print(f'{indent}isEncoded: {self.isEncoded}')
        print(f'{indent}isTrainTestSplit: {self.isTrainTestSplit}')
        print(f'{indent}isImportantFeaturesApplied: {self.isImportantFeaturesApplied}')
        



    def displayClassBalance(self, columnName=None, verbose=False, showRecords=5):
        if columnName is None:
            columnName = self.targetColumn

        dps.displayClassBalance(data=self.getWorkingData(),
                                columnName=columnName,
                                showRecords=showRecords,
                                verbose=verbose)

    def classBalanceUndersample(self,
                                sampleSize=None,
                                columnName=None):

        if columnName is None:
            columnName = self.targetColumn

        # Needs to be balanced
        dfBalanced = dps.classBalanceUndersample(dataFrame=self.getWorkingData(),
                                                 columnName=columnName,
                                                 sampleSize=sampleSize)

        if not self.isBalanced:
            self.__setWorkingData(dfBalanced)
            self.isBalanced = True

    # Class balance oversample will oversample for training data only
    # Must have been train/test split prior to this running
    def classBalanceOversample(self,
                               verbose=True,
                               show_records=5,
                               random_state=765):
        
        if not self.isTrainTestSplit:
            print(f'Data has not been train_test_split.')
            print(f'Oversampling will only occur on training data')
            return
        else:
            print(f'Data has been train/test split')
            print(f'Oversampling for class balance will occur on the training datset only')
        
        retDF = dps.classBalanceOversample(dataFrame=self.getTrainData(),
                                          columnName=self.targetColumn,
                                          random_state=random_state,
                                          verbose=verbose,
                                          show_records=show_records)
        
        
        if not self.isBalanced:
            self.__setTrainData(retDF)
            self.isBalanced = True
        
        

    def getXTrainData(self,
                      finalFeatures=None):

        if finalFeatures is None:
            useFeatures = self.dataFeatures
        else:
            useFeatures = finalFeatures

        return self.getTrainData()[useFeatures]

    def getXTestData(self,
                     finalFeatures=None):

        if finalFeatures is None:
            useFeatures = self.dataFeatures
        else:
            useFeatures = finalFeatures

        return self.getTestData()[useFeatures]

    def getYTrainData(self):
        return self.getTrainData()[self.targetColumn]

    def getYTestData(self):
        return self.getTestData()[self.targetColumn]

    def save(self,
             filename):
        if filename is None:
            print(f'No filename provided. Please provide full path')

        with open(filename, 'wb') as f:
            print(f'Saving file as {filename}')
            pickle.dump(self, f)


    def set_important_features(self, keep_features):
        # Needs to be set for working/train/test datasets
        
        print(f'Filtering data for only important features')
        if self.isWorkingDataLoaded:
            self.workingData = self.workingData[keep_features]
            
        if self.isTrainDataLoaded:
            self.trainData = self.trainData[keep_features]
            
        if self.isTestDataLoaded:
            self.testData = self.testData[keep_features]
            
        self.setDataFeatures(feature_list=self.workingData.columns)
        self.isImportantFeaturesApplied = True
        

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Holds all relevant processing/cleaning parameters
@dataclass        
class DataPackageParams:
    __version = 0.2

    

    # Text Cleaning Params
    fix_unicode: bool = True  # fix various unicode errors
    to_ascii: bool = True  # transliterate to closest ASCII representation
    lower: bool = True  # lowercase text
    no_line_breaks: bool = False  # fully strip line breaks as opposed to only normalizing them
    no_urls: bool = False  # replace all URLs with a special token
    no_emails: bool = False  # replace all email addresses with a special token
    no_phone_numbers: bool = False  # replace all phone numbers with a special token
    no_numbers: bool = False  # replace all numbers with a special token
    no_digits: bool = False  # replace all digits with a special token
    no_currency_symbols: bool = False  # replace all currency symbols with a special token
    no_punct: bool = False  # remove punctuations
    replace_with_punct: str = ""  # instead of removing punctuations you may replace them
    replace_with_url: str = "<URL>"
    replace_with_email: str = "<EMAIL>"
    replace_with_phone_number: str = "<PHONE>"
    replace_with_number: str = "<NUMBER>"
    replace_with_digit: str = "0"
    replace_with_currency_symbol: str = "<CUR>"
    lang: str = "en"  # set to 'de' for German special handling

    # Contractions
    fix_contractions: bool = False

    # Lemmatize
    lemmatize: bool = False
    
    #Remove small tokens
    remove_small_tokens: bool = False
    min_token_size: int = 0

    # Stopwords
    remove_stopwords: bool = True  # Remove stopwords
    stopword_language: str = 'english'
    custom_stopwords: list = None

    # Class Balance
    balance_dataset: bool = False
    balance_type: str = 'undersample'
    undersample_size: int = None  # Can be set to an absolute value. None means undersample to smallest
    
    # train test split params
    stratifyColumn: str = None
    train_size: float = 0.8
    random_state: int = 765
    shuffle: bool = True

    # Encoding params
    encoding_type: str = 'TFIDF'
    max_features: int = 100  # Currently only used in TFIDF
   
    
    def save(self,
             filename):
        if filename is None:
            print(f'No filename provided. Please provide full path')

        with open(filename, 'wb') as f:
            print(f'Saving file as {filename}')
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
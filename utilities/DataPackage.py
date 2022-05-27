import DataPackageSupport as dps
import pickle
import gzip

class DataPackage:
    __version = 0.2

    def __init__(self,
                 original_data,
                 unique_column,
                 target_column,
                 data_column,
                 data_package_params=None):
        self.isProcessed = False
        # TODO Needs to be updated to deal with unsupervised
        self.targetColumn = target_column

        self.__setOrigData(original_data, unique_column)

        self.dataColumn = data_column



        if data_package_params is None:
            self.hasDataPackageParams = False
        else:
            self.__setDataPackageParams(data_package_params=data_package_params)

    # Set the original dataFrame (pandas)
    def __setOrigData(self, origData, unique_column):
        self.origData = origData
        self.__setUniqueColumn(unique_column)
        self.isOrigDataLoaded = True

        # Keep orig data untainted for process/integrity
        # WorkingData is what we will use for processing
        self.__setWorkingData(origData=origData)

        # A new dataframe means we need to reset our work
        self.__resetWork()

    # Set working data. Leaves original dataframe intact
    def __setWorkingData(self,
                         origData):
        self.workingData = origData.copy()
        self.isWorkingDataLoaded = True

        #Get and set features listing, removing unique and target columns
        self.dataFeatures = list(self.workingData.columns)
        # Remove unique and target columnm
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

        # Check to see if we need to process it
        if self.data_package_params.process_params:
            print(f'Params loaded. Process set to True.')
            print(f'Processing Data Package')
            self.__processDataPackage()

    def processDataPackage(self):
        self.data_package_params.process_params = True
        self.__processDataPackage()

    # Process the datapackage. Called from __setDataPackageParams
    def __processDataPackage(self):
        # Confirm it hasn't already been processed
        if self.isProcessed:
            print(f'DataPackage has already been processed')
            return

        # Confirm we are to process it
        if self.data_package_params.process_params:
            self.display()
            print(f'Processing data package with provided parameters')

            # classbalance undersample
            self.classBalanceUndersample(sampleSize=self.data_package_params.sample_size)

            # process text column
            self.clean_text_column(data_package_params=self.data_package_params)

            # remove stopwords
            self.remove_stopwords(stopword_language=self.data_package_params.stopword_language)

            # process_TFIDF
            self.process_TFIDF(max_features=self.data_package_params.max_features)

            # splittraintest
            self.splitTrainTest(stratifyColumn=self.data_package_params.stratifyColumn,
                                train_size=self.data_package_params.train_size,
                                random_state=self.data_package_params.random_state,
                                shuffle=self.data_package_params.shuffle)
            print(f'')
            print(f'Processing data package has been completed')
            print(f'')
            self.isProcessed = True
            self.display()

        else:
            print(f'__processDataPackage called with process_params=False')
            return

    # Function for cleaning text, lemmatization, etc
    # Basic functionality included now, to be expanded later
    # Goal is to keep this to one function if possible
    def clean_text_column(self,
                          data_package_params=None):
        dps.clean_text_column(dataFrame=self.workingData,
                              data_column=self.dataColumn,
                              data_package_params=data_package_params)

        self.isCleaned = True

    # Remove stopwords
    def remove_stopwords(self, stopword_language='english'):
        dps.remove_stopwords(dataFrame=self.workingData,
                             data_column=self.dataColumn,
                             stopword_language='english'
                             )
        self.isStopWorded = True

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
        return self.trainData

    def __setTestData(self, testData):
        self.testData = testData
        self.isTestDataLoaded = True

    def getTestData(self):
        return self.testData

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
        if self.isOrigDataLoaded == False:
            display("Original data frame is not loaded")
        return self.origData

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

        print(f'{emptySpace}Original Data:')
        print(f'{indent}original data shape: {self.origData.shape}')
        print(f'{emptySpace}Working Data:')
        print(f'{indent}working data shape: {self.workingData.shape}')

        print(f'{emptySpace}Process:')
        print(f'{indent}isProcessed: {self.isProcessed}')
        print(f'{indent}isCleaned: {self.isCleaned}')
        print(f'{indent}isStopWorded: {self.isStopWorded}')
        print(f'{indent}isBalanced: {self.isBalanced}')
        print(f'{indent}isEncoded: {self.isEncoded}')
        print(f'{indent}isTrainTestSplit: {self.isTrainTestSplit}')

        print(f'{emptySpace}Data:')
        print(f'{indent}isOrigDataLoaded: {self.isOrigDataLoaded}')
        print(f'{indent}isTrainDataLoaded: {self.isTrainDataLoaded}')
        print(f'{indent}isTestDataLoaded: {self.isTrainDataLoaded}')
        print(f'')

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

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)



class DataPackageParams:
    __version = 0.1

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

    def __init__(self,
                 # Process package on load?
                 process_params=True, # True to process package on load

                 # Class Balance
                 sample_size=None,  # Can be set to an absolute value. None means undersample to smallest

                 # Text Cleaning Params
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 lower=True,  # lowercase text
                 no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
                 no_urls=False,  # replace all URLs with a special token
                 no_emails=False,  # replace all email addresses with a special token
                 no_phone_numbers=False,  # replace all phone numbers with a special token
                 no_numbers=False,  # replace all numbers with a special token
                 no_digits=False,  # replace all digits with a special token
                 no_currency_symbols=False,  # replace all currency symbols with a special token
                 no_punct=False,  # remove punctuations
                 replace_with_punct="",  # instead of removing punctuations you may replace them
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="0",
                 replace_with_currency_symbol="<CUR>",
                 lang="en",  # set to 'de' for German special handling

                 # Stopwords
                 remove_stopwords=True,  # Remove stopwords
                 stopword_language='english',

                 # train test split params
                 stratifyColumn=None,
                 train_size=0.8,
                 random_state=765,
                 shuffle=True,

                 # Encoding params
                 encoding_type='TFIDF',
                 max_features=100 # Currently only used in TFIDF
                 ):

        # Process package
        self.process_params = process_params

        # Class Balance
        self.sample_size = sample_size  # Can be set to an absolute value. None means undersample to smallest

        # Stopwords
        self.remove_stopwords = remove_stopwords
        self.stopword_language = stopword_language

        # Balance params
        self.fix_unicode = fix_unicode  # fix various unicode errors
        self.to_ascii = to_ascii  # transliterate to closest ASCII representation
        self.lower = lower  # lowercase text
        self.no_line_breaks = no_line_breaks  # fully strip line breaks as opposed to only normalizing them
        self.no_urls = no_urls  # replace all URLs with a special token
        self.no_emails = no_emails  # replace all email addresses with a special token
        self.no_phone_numbers = no_phone_numbers  # replace all phone numbers with a special token
        self.no_numbers = no_numbers  # replace all numbers with a special token
        self.no_digits = no_digits  # replace all digits with a special token
        self.no_currency_symbols = no_currency_symbols  # replace all currency symbols with a special token
        self.no_punct = no_punct  # remove punctuations
        self.replace_with_punct = replace_with_punct  # instead of removing punctuations you may replace them
        self.replace_with_url = replace_with_url
        self.replace_with_email = replace_with_email
        self.replace_with_phone_number = replace_with_phone_number
        self.replace_with_number = replace_with_number
        self.replace_with_digit = replace_with_digit
        self.replace_with_currency_symbol = replace_with_currency_symbol
        self.lang = lang  # set to 'de' for German special handling

        # Train/Test Split params
        self.stratifyColumn = stratifyColumn
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle


        # Encoding params
        self.encoding_type = encoding_type
        self.max_features = max_features



import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool

def exploreDataframe(data, showRecords=5):
    myDF = data.copy()
    print("dataframe shape: " + str(myDF.shape))
    print("\ndataframe info: ")
    print(myDF.info())
    print(f'')
    print(f'Null value count by column:')
    display(myDF.isnull().sum())
    print(f'')
    print("\nFirst " + str(showRecords) + " in dataframe")
    display(myDF.head(showRecords))
    print("\nLast " + str(showRecords) + " in dataframe")
    display(myDF.tail(showRecords))


def showUniqueColVals(dataFrame,
                      colName,
                      showRecords=5):
    dataTypeObj = dataFrame.dtypes[colName]
    numRecords = len(dataFrame)
    uniqueItems = dataFrame[colName].unique()
    numUnique = len(uniqueItems)
    nullValues = dataFrame[colName].isna().sum()
    if numRecords == 0:
        percUnique = 0
    else: 
        percUnique = str(round(numUnique / numRecords, 3) * 100 ) + "%"

    print(f'Data type of column [{colName}] is: {dataTypeObj}')
    print(f'Total number of rows: {numRecords}')

    print(f'Unique values in column: {numUnique} [percent unique: {percUnique}]')
    print(f'Null values in column: {nullValues}')
    print(f'List of unique values:')
    # print(sort(uniqueItems)) # not all series are sortable apparently
    print(uniqueItems)

    sumDF = dataFrame.groupby([colName]).size().to_frame('record_count')
    sumDF.reset_index(inplace=True)

    print(f'')
    print(f'Top {showRecords} records by frequency for {colName}')
    sumDF = sumDF.sort_values(by=['record_count'], ascending=False)
    print(sumDF.head(showRecords))
    topList = sumDF[colName].head(showRecords).tolist()



    print(f'')
    print(f'Bottom {showRecords} records by frequency for {colName}')
    sumDF = sumDF.sort_values(by=['record_count'], ascending=True)
    print(sumDF.head(showRecords))
    bottomList = sumDF[colName].head(showRecords).tolist()

    return topList, bottomList


def dropNullRows(dataFrame, columns=None, showRecords=1):
    if columns is None:
        print(f'Dropping rows where any column is null')
    else:
        print(f'Dropping rows where {columns} is/are null')

    print(f'')
    print(f'Original dataFrame shape: {dataFrame.shape}')
    print(f'Original null value count by column:')
    display(dataFrame.isna().sum())
    if columns is None or len(columns) == 0:
        tDF = dataFrame.dropna(axis=0, subset=None)
    else: # Specified some sort of column
        tDF = dataFrame.dropna(axis=0, subset=None)

    print(f'')
    print(f'*** Rows with nulls meeting criteria have been dropped')
    print(f'')
    print(f'New values:')
    exploreDataframe(tDF, showRecords=1)
    print(f'New dataframe returned')
    return tDF


def show_column_text_length_summary(dataFrame, column):
    lengthCol = 'Length of text'
    print(f'Displaying length summary for column: {column}')
    myDF = dataFrame.copy()
    myDF[lengthCol] = myDF[column].str.len()
    print(myDF[lengthCol].describe().apply(lambda x: format(x, '.0f')))

def setPlotSize2(plot_w, plot_h):
    sns.set(rc={'figure.figsize': (plot_w, plot_h)})

def setPlotSize(plotsize):
    if plotsize == 7:
        sns.set(rc={'figure.figsize': (15, 30)})
    elif plotsize == 6:
        sns.set(rc={'figure.figsize': (20, 15)})
    elif plotsize == 5:
        sns.set(rc={'figure.figsize': (20, 8)})
    elif plotsize == 4:
        sns.set(rc={'figure.figsize': (5, 25)})
    elif plotsize == 3:
        sns.set(rc={'figure.figsize': (5, 20)})
    elif plotsize == 2:
        sns.set(rc={'figure.figsize': (5, 15)})
    elif plotsize == 1:
        sns.set(rc={'figure.figsize': (5, 10)})
    else:  # Should be size 1
        # should only be one but catch it and default to size 1
        sns.set(rc={'figure.figsize': (5, 5)})    

# Goal is to pad column ID in sampleDF so it has all missing values
# between min (set at 0) and maxPad
def padDF(dataFrame, 
          columnToPad, 
          valueColumn, 
          setValue=None, 
          binsize=1,
          maxPad=None):
    
    if maxPad is None:
        maxValue = dataFrame[columnToPad].max()
    else: 
        maxValue = maxPad
    
    # Create list of possible ranges
    fullValueList = []
    for x in range(0,maxValue+1, binsize):
        fullValueList.append(x)
    
    # Get values that exist
    existingValueList = dataFrame[columnToPad].unique().tolist()
    
    fullValueSet = set(fullValueList)
    existingSet = set(existingValueList)
    
    addValueList = list(fullValueSet.difference(existingSet))
    
    appendList = []
    # Create new list to append to original dataFrame
    for row in addValueList:
        appendList.append([row, setValue])
        
    # Turn it into a dataframe
    tDf = pd.DataFrame (appendList, columns = [columnToPad, valueColumn])
    
    tDf = tDf.append(dataFrame, ignore_index=True)
    tDf.sort_values(by=[columnToPad], inplace=True)
    tDf.reset_index(drop=True, inplace=True)
    return tDf


# Takes initial dataframe, summarizes (group by and count) to check values
# Offers features for zoom and truncation
def getAnalyzedFrame(df,
                     colName,
                     binsize=5,
                     maxPad=None,
                     verbose=False,
                     zoom=False,
                     minZoomLevel=0,
                     maxZoomLevel=0,
                     plotsize=6,
                     numRecords=2):
    
    binColName = f'bin_at_{str(binsize)}'
    binnedCountName = 'binnedCount'

    # Parameter checking
    if (binsize <= 0):
        print(f'binsize of {str(binsize)} given. Must be > 0 and evenly divisible by 10 or = 1')
        return

    if zoom:
        if maxZoomLevel < minZoomLevel:
            print(f'maxZoomLevel given as {str(maxZoomLevel)} which must ' +
                  f'be >= minZoomLevel given as {str(minZoomLevel)}')
            return

    # Make a copy of the incoming frame as we will be manipulating it
    tDf = df.copy()
    

    #Null values make it flunk out.
    numNullValues = tDf[colName].isnull().sum()
    if numNullValues > 0:
        print(f'Warning: {numNullValues} null values detected in column. Removing for analysis')
        tDf = tDf.dropna(subset=[colName], axis=0)

    #Groupby and summarize dataframe
    tDf = round(tDf[[colName]], 0).astype(int)
    tDf[binColName] = [int(math.trunc(val / binsize) * binsize) for val in tDf[colName]]

    tDf = tDf.groupby(binColName).size().to_frame(binnedCountName).sort_values([binColName], ascending=False)
    tDf.reset_index(inplace=True)
    tDf = padDF(dataFrame=tDf,
                columnToPad=binColName,
                valueColumn=binnedCountName,
                setValue=0,
                binsize=binsize,
                maxPad=maxPad
                )
    #display(tDf.head(10))
    #Zoom to applicable level
    if zoom:
        tDf = tDf.loc[(tDf[binColName] >= minZoomLevel) & (tDf[binColName] <= maxZoomLevel)]
        tDf.reset_index(drop=True, inplace=True)

    if verbose:
        exploreDataframe(tDf,showRecords=numRecords)
    
    return tDf, binColName, binnedCountName

# Analyze text column to show character/alpha/numeric/space composition
def analyzeTextColumn(dataFrame, 
                      column, 
                      binsize=1,
                      zoom=False,
                      minZoomLevel=0,
                      maxZoomLevel=0,
                      num_cores=4,
                      plotsize=7):
    
    myDF = dataFrame.copy()
    print(f'Analyzing column: {column}')
    
    tokenColName = 'token_count'
    lengthColName = 'length_text'
    num_alpha = 'num_alpha'
    digitCountColName = 'num_digits'
    nonAlphaCountColName = 'num_nonAlphaNumeric'
    
    analysisColumns = [tokenColName,
                       lengthColName,
                       num_alpha,
                       digitCountColName,
                       nonAlphaCountColName]
    
    indent=f'-->'
    
    print(f'Column description')
    print(f'{indent}column {tokenColName}: Number of tokens')
    print(f'{indent}column {lengthColName}: Number of non-whitespace characters')
    print(f'{indent}column {num_alpha}: Number of alpha characters. Case insensitive')
    print(f'{indent}column {digitCountColName}: Number of digits in string. 123 counts as 3 digits')
    print(f'{indent}column {nonAlphaCountColName}: Number of non-alphanumeric characters. abc123abc counts as 3')

    print(f'')
    print(f'Generating metrics with {num_cores} processes...', end='')
    # Stores parameters for sending to function to analyze frame
    func_input = []

    df_split = np.array_split(myDF, num_cores)
    for count, df_part in enumerate(df_split, start=1):
        func_input.append([df_part, column, tokenColName, lengthColName, num_alpha, digitCountColName,nonAlphaCountColName,count])

    pool = Pool(num_cores)
    myDF = pd.concat(pool.starmap(starAnalyze, func_input))
    pool.close()
    pool.join()
   
    print(f'completed.')
    # Get analyzed dataframe for each column to plot
    analyzedFrames = []
    tDf = None
    binColName = None
    binnedCountName = None
    
    # Check the maximum value for each column
    # We want to pad everything so X-axis values are aligned 
    # across charts
    maxValue = 0
    for col in analysisColumns:
        maxValue = max(maxValue, myDF[col].max())
        
    #print(f'MaxValue found of: {maxValue}')
    for col in analysisColumns:
        print(f'Summarizing data for: {col}...', end='')
        tDf, binColName, binnedCountName = getAnalyzedFrame(df=myDF,
                                                            colName=col,
                                                            binsize=binsize,
                                                            maxPad=maxValue,
                                                            zoom=zoom,
                                                            minZoomLevel=minZoomLevel,
                                                            maxZoomLevel=maxZoomLevel)
        print(f'completed')
        analyzedFrames.append([col, binColName, binnedCountName, tDf])
    
    
    
    plt.close()
    setPlotSize(plotsize)
    plt.suptitle(f'Text analysis of column {column}', fontsize=14)
    
    
    for n, analysisInfo in enumerate(analyzedFrames):    
        print(f'Adding plot for: {analysisInfo[0]}')
        
        # add a new subplot iteratively
        ax = plt.subplot(5, 1, n + 1)
        
        # filter df and plot ticker on the new subplot axis
        
        max_y = int(analysisInfo[3][binnedCountName].max())+1
        
        sns.barplot(x=analysisInfo[1], 
                    y=analysisInfo[2], 
                    data=analysisInfo[3],
                    color='b')
    
        plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90-degrees
        # chart formatting
        #ax.set_title(f'Title: n={n}')
        
        #plt.yticks(np.arange(0, max_y, 1))
        ax.set_xlabel(analysisInfo[0])
        ax.set_ylabel('Document count')
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.3)
    
    plt.tight_layout()
    plt.show()
    

# Analyze text column to show character/alpha/numeric/space composition
def starAnalyze(dataFrame, 
                 column,
                 tokenColName,
                 lengthColName,
                 num_alpha,
                 digitCountColName,
                 nonAlphaCountColName, 
                 processingNum):
     
    # Create column for text length
    dataFrame[lengthColName] = dataFrame[column].str.findall(r'[^ ]').str.len()
    
    # Create column for num tokens
    dataFrame[tokenColName] = dataFrame[column].str.replace(',','').str.split().str.len()
    
    # Create column for count of DIGITS in 
    dataFrame[digitCountColName] = dataFrame[column].str.findall(r'[0-9]').str.len()
    
    
    # Count alpha characters
    dataFrame[num_alpha] = dataFrame[column].str.findall(r'[a-zA-Z]').str.len()
    
    
    # Count non-alphanumeric characters
    dataFrame[nonAlphaCountColName] = dataFrame[column].str.findall(r'[^a-zA-Z0-9 ]').str.len()
    
    return dataFrame
import numpy as np

def exploreDataframe(data, showRecords=5):
    print("dataframe shape: " + str(data.shape))
    print("\ndataframe info: ")
    print(data.info())
    print("\nFirst " + str(showRecords) + " in dataframe")
    display(data.head(showRecords))
    print("\nLast " + str(showRecords) + " in dataframe")
    display(data.tail(showRecords))

    print(f'')
    print(f'Null value count by column:')
    display(data.isna().sum())


def showUniqueColVals(dataFrame,
                      colName,
                      showRecords=5):
    dataTypeObj = dataFrame.dtypes[colName]
    numRecords = len(dataFrame)
    uniqueItems = dataFrame[colName].unique()
    numUnique = len(uniqueItems)
    nullValues = dataFrame[colName].isna().sum()
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

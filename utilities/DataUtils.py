import numpy as np

def exploreDataframe(data, numRecords=5):
    print("dataframe shape: " + str(data.shape))
    print("\ndataframe info: ")
    print(data.info())
    print("\nFirst " + str(numRecords) + " in dataframe")
    display(data.head(numRecords))
    print("\nLast " + str(numRecords) + " in dataframe")
    display(data.tail(numRecords))

def showUniqueColVals(dataFrame,
                      colName,
                      showRecords=5):
    dataTypeObj = dataFrame.dtypes[colName]
    print(f'Data type of column [{colName}] is: {dataTypeObj}')
    print(f'Total number of rows: {len(dataFrame)}')
    a = dataFrame[colName].unique()
    print(f'Unique values: {len(a)}')
    print(f'List of unique values:')
    # print(sort(a)) # not all series are sortable apparently
    print(a)

    sumDF = dataFrame.groupby([colName]).size().to_frame('record_count')
    sumDF.reset_index(inplace=True)

    print(f'')
    print(f'Showing top {showRecords} records for {colName} by frequency')
    sumDF = sumDF.sort_values(by=['record_count'], ascending=False)
    print(sumDF.head(showRecords))

    print(f'')
    print(f'Showing bottom {showRecords} records for {colName} by frequency')
    sumDF = sumDF.sort_values(by=['record_count'], ascending=True)
    print(sumDF.head(showRecords))

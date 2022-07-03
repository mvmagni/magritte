import numpy as np

def exploreDataframe(data, numRecords=5):
    print("dataframe shape: " + str(data.shape))
    print("\ndataframe info: ")
    print(data.info())
    print("\nFirst " + str(numRecords) + " in dataframe")
    display(data.head(numRecords))
    print("\nLast " + str(numRecords) + " in dataframe")
    display(data.tail(numRecords))

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
    percUnique = str(round(numUnique / numRecords, 2) * 100 ) + "%"

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

    print(f'')
    print(f'Bottom {showRecords} records by frequency for {colName}')
    sumDF = sumDF.sort_values(by=['record_count'], ascending=True)
    print(sumDF.head(showRecords))

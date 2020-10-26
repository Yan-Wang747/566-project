
import os
import pathlib
import pandas as pd
import shared

_pathToData = pathlib.Path(__file__).parent.joinpath('data')

def __loadCSV(fileName: pathlib.Path):
    
    df = pd.read_csv(
        fileName,
        header=None,
        names=shared.CSV_COLUMNS
    )
    
    # handle invalid data
    idsToRemove = set()

    for rowOfpound in list(df[df.astype(str)['id'] == '#'].index):
        if rowOfpound == 0:
            continue

        idsToRemove.add(df.iloc[rowOfpound - 1]['id'])

    for i in df.id.unique():
        if len(df[df['id'] == i]) < 25:
            idsToRemove.add(i)

    for id in idsToRemove:
        df = df[df['id'] != id]

    return df

def __loadSubject(subjectName: str):

    pathToSubjectFolder = _pathToData.joinpath(subjectName)

    dfs = []
    for fileName in os.listdir(pathToSubjectFolder):
        if '.csv' not in fileName:
                continue

        label = fileName.replace('.csv', '')
        assert label in shared.LABELS or label == shared.CALI_NAME

        df = __loadCSV(pathToSubjectFolder.joinpath(fileName))
 
        df['label'] = label
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def loadSubjects(subjects):
    '''
    load csvs of all subdirs
    return one single pandas dataframe
    '''
    dfs = {}

    for subject in subjects:
        dfs[subject] = __loadSubject(subject)

    return dfs

if __name__ == "__main__":
    df = __loadSubject('kelly_11_7')
    print(df)
    print(len(df))

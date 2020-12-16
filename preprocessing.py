from torch._C import dtype
import shared
import numpy as np
import raw_data_reader

from augment import augment
from scipy import interpolate

def __calibrate(dfs):
    subjectsDataset = {}
    for subject, df in dfs.items():
        calibrationFrame = df[df['label'] == shared.CALI_NAME]

        assert not calibrationFrame.empty
        
        csvColumnsWO_id_td = [ name for name in calibrationFrame.columns if name in shared.CSV_COLUMNS and name not in ['id', 'td'] ]
        calibrationArray = calibrationFrame[csvColumnsWO_id_td].to_numpy() # don't average the id and td

        calibrationVec = np.mean(calibrationArray, axis=0) 
        
        dataset = {}
        labels = df['label'].unique()
        for label in labels:
            if label == shared.CALI_NAME:
                continue

            dataWithLabel = df[df['label'] == label]
            ids = dataWithLabel.id.unique()

            dataSequences = []
            for id in ids:
                dataWithID = dataWithLabel[dataWithLabel['id'] == id]
                pureData = dataWithID[csvColumnsWO_id_td].to_numpy()

                # apply calibration
                pureData -= calibrationVec + pureData[0]
                id_td = dataWithID[['id', 'td']].to_numpy()
                dataSequences.append(np.concatenate((id_td, pureData), axis=1))

            dataset[label] = dataSequences

            subjectsDataset[subject] = dataset
    
    return subjectsDataset

def __createTimeStamps(tdVec):
    timeStamps = tdVec
    for i in range(1, len(timeStamps)):
        timeStamps[i] += timeStamps[i-1]
    
    return timeStamps

def __interpolateSeq(sequence, numOfPoints):
    
    timeStamps = __createTimeStamps(sequence[:, 1])
    timeLowerBound = timeStamps[0]
    timeUpperBound = timeStamps[-1]
    timeAxis = np.linspace(timeLowerBound, timeUpperBound, num=numOfPoints)

    interpolatedSeq = [[sequence[0, 0]]*numOfPoints, timeAxis]
    for col in range(2, len(sequence[0])):
        interploateFunc = interpolate.interp1d(timeStamps, sequence[:, col])
        interpolatedSeq.append(interploateFunc(timeAxis))

    return np.column_stack(interpolatedSeq)

def __interpolateDataSet(subjectsDataDict):
    interpolatedDataDict = {}

    for subject, dataset in subjectsDataDict.items():
        interpolated = {}
        for label, dataSequences in dataset.items():
            newSequences = [ __interpolateSeq(sequence, shared.NUM_OF_INTERP_POINTS) for sequence in dataSequences ]
            interpolated[label] = newSequences
        
        interpolatedDataDict[subject] = interpolated

    return interpolatedDataDict

def __selectColumns(dfs, columns):
    new_dfs = {}
    columns = ['id', 'td'] + columns + ['label'] # order matters

    for subject, df in dfs.items():
        new_dfs[subject] = df[columns]

    return new_dfs

def __splitSubjects(subjects, validationRatio, testRatio):

    numOfVal = int(len(subjects) * validationRatio)
    numOfTest = int(len(subjects) * testRatio)
    numOfTrain = len(subjects) - numOfVal - numOfTest

    shuffledSubjects = np.random.permutation(subjects)

    trainingSubjects = shuffledSubjects[:numOfTrain]

    validationSubjects = shuffledSubjects[numOfTrain:numOfTrain+numOfVal]
        
    testSubjects = shuffledSubjects[numOfTrain+numOfVal:]

    return trainingSubjects, validationSubjects, testSubjects

def __toDataset(subjectsDataDict):
    xs = []
    labels = []

    for _, data in subjectsDataDict.items():
        for label, samples in data.items():
            labelIndex = shared.LABELS.index(label)

            for sample in samples:
                xs.append(sample[:, 2:]) # discard id, td
                labels.append(labelIndex)
    
    return np.array(xs, dtype=np.float32), np.array(labels, dtype=np.long)

def __augmentTrainingSet(trainingX, trainingLables, augmentProp):
    aug_xs = []
    aug_labels = []

    for _ in range(augmentProp):
        for i in range(trainingX.shape[0]):
            x = trainingX[i]
            y = trainingLables[i]

            augmented_x = augment(x)

            aug_xs.append(augmented_x)
            aug_labels.append(y)
    
    return np.array(trainingX.tolist() + aug_xs), np.array(trainingLables.tolist() + aug_labels)

def __normalize(trainingX, validationX, testX):
    xs = trainingX
    if len(validationX) > 0:
        xs = np.concatenate((xs, validationX), axis=0)

    if len(testX) > 0:
        xs = np.concatenate((xs, testX), axis=0)

    xs = (xs - np.mean(xs, axis=0)) / np.std(xs, axis=0)

    return xs[:len(trainingX)], xs[len(trainingX):len(trainingX)+len(validationX)], xs[len(trainingX)+len(validationX):]

def __denoise(xs, n):
    xs_denoised = np.cumsum(xs, axis=0)
    xs_denoised[n:] = xs_denoised[n:] - xs_denoised[:-n]
    
    return xs_denoised / n

def loadData(subjects=shared.SUBJECTS,
             selectedColumns=['yaw', 'pitch', 'roll'],  
             splitMode=shared.SPLIT_MODE_CLASSIC,
             validationRatio=0.2,
             testRatio=0.2,
             flatten=False,
             normalize=True,
             denoise_n=8):

    trainingX = None
    trainingLabels = None
    validationX = []
    validationLabels = []
    testX = []
    testLabels = []

    if splitMode == shared.SPLIT_MODE_CLASSIC:
        subjectsDataDict = raw_data_reader.loadSubjects(subjects)
        subjectsDataDict = __selectColumns(subjectsDataDict, selectedColumns)
        subjectsDataDict = __calibrate(subjectsDataDict)
        subjectsDataDict = __interpolateDataSet(subjectsDataDict)
        xs, labels = __toDataset(subjectsDataDict)
        xs = __denoise(xs, denoise_n)

        perm = np.random.permutation(range(xs.shape[0]))
        xs = xs[perm]
        labels = labels[perm]

        numOfVal = int(xs.shape[0]*validationRatio)
        numOfTest = int(xs.shape[0]*testRatio)
        numOfTrain = len(xs) - numOfVal - numOfTest

        trainingX = xs[:numOfTrain]
        trainingLabels = labels[:numOfTrain]

        # trainingX, trainingLabels = __augmentTrainingSet(trainingX, trainingLabels, augmentProp)

        validationX = xs[numOfTrain:numOfTrain+numOfVal]
        validationLabels = labels[numOfTrain:numOfTrain+numOfVal]
            
        testX = xs[numOfTrain+numOfVal:]
        testLabels = labels[numOfTrain+numOfVal:]
        
    elif splitMode == shared.SPLIT_MODE_BY_SUBJECT:
        trainingSubjects, validationSubjects, testSubjects = __splitSubjects(subjects, validationRatio, testRatio)
        subjectsDataDict = raw_data_reader.loadSubjects(trainingSubjects)
        subjectsDataDict = __selectColumns(subjectsDataDict, selectedColumns)
        subjectsDataDict = __calibrate(subjectsDataDict)
        subjectsDataDict = __interpolateDataSet(subjectsDataDict)
        trainingX, trainingLabels = __toDataset(subjectsDataDict)
        trainingX = __denoise(trainingX, denoise_n)
        # trainingX, trainingLabels = __augmentTrainingSet(trainingX, trainingLabels, augmentProp)

        if len(validationSubjects) > 0:
            subjectsDataDict = raw_data_reader.loadSubjects(validationSubjects)
            subjectsDataDict = __selectColumns(subjectsDataDict, selectedColumns)
            subjectsDataDict = __calibrate(subjectsDataDict)
            subjectsDataDict = __interpolateDataSet(subjectsDataDict)
            validationX, validationLabels = __toDataset(subjectsDataDict)
            validationX = __denoise(validationX, denoise_n)
        if len(testSubjects) > 0:
            subjectsDataDict = raw_data_reader.loadSubjects(testSubjects)
            subjectsDataDict = __selectColumns(subjectsDataDict, selectedColumns)
            subjectsDataDict = __calibrate(subjectsDataDict)
            subjectsDataDict = __interpolateDataSet(subjectsDataDict)
            testX, testLabels = __toDataset(subjectsDataDict)
            testX = __denoise(testX, denoise_n)

    if flatten:
        trainingX = np.squeeze(np.reshape(trainingX, (trainingX.shape[0], 1, -1)))
        if len(validationX) > 0:
            validationX = np.squeeze(np.reshape(validationX, (validationX.shape[0], 1, -1)))
        if len(testX) > 0:
            testX = np.squeeze(np.reshape(testX, (testX.shape[0], 1, -1)))

    if normalize:
        trainingX, validationX, testX = __normalize(trainingX, validationX, testX)
    
    return np.array(trainingX, dtype=np.float32), np.array(trainingLabels, dtype=np.long), \
           np.array(validationX, dtype=np.float32), np.array(validationLabels, dtype=np.long), \
           np.array(testX, dtype=np.float32), np.array(testLabels, dtype=np.long)

if __name__ == '__main__':
    loadData(flatten=True)
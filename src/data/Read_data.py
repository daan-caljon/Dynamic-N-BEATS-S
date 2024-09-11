import pandas as pd
import numpy as np

def read_data(dataset):
    if dataset == "M3_Monthly":
        df = pd.read_csv(r"")

        mydata = list()
        temp_list = list()
        my_id = "M1" #first ID
        for num, row in df.iterrows():
            if my_id == row["series_id"]:
                temp_list.append(row["value"])
            else:
                my_id = row["series_id"]
                mydata.append(np.array(temp_list))
                temp_list = list()
                temp_list.append(row["value"])
                
        mydata.append(np.array(temp_list)) #append the last one
        dataset_clean = [x[x == x] for x in mydata]        

        testset_m3m = dataset_clean.copy() 
        valset_m3m = [x[:-18] for x in testset_m3m] 
        trainset_m3m = [x[:-18] for x in valset_m3m]
        return trainset_m3m, valset_m3m, testset_m3m

    elif dataset == "M4_Monthly":
        # Load monthly M4 data.
        # Transform the data into a lists of arrays. Each inner array represents a timeseries.
        # Remove all the NaN values from the datasets.

        # M4
        trainset = pd.read_csv('https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv')
        testset = pd.read_csv('https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv')
        trainset.set_index('V1', inplace = True)
        testset.set_index('V1', inplace = True)
        # Add the testset columns behind the trainset columns
        testset_merge = trainset.merge(testset, on = 'V1', how = 'inner')
        # Get the data in numpy representation
        trainset_np = trainset.values
        testset_np = testset_merge.values
        # Select all non NaN values from the trainset
        trainset_clean = [x[x == x] for x in trainset_np]
        # Train/validation/test --------------------------------- NBeats paper validation strategy
        testset_m4m = [x[x == x] for x in testset_np]
        valset_m4m = trainset_clean.copy()
        trainset_m4m = [x[:-18] for x in trainset_clean]


        del(trainset, testset, testset_merge, trainset_np, testset_np, trainset_clean)
        return trainset_m4m, valset_m4m, testset_m4m
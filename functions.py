def convert_to_deg(px, im_size):
    delta = px - im_size/2
    ecc = delta*(8/im_size)
    return ecc 

def regression_indices(number_of_data_values,number_of_groups,test_group,SEED):
    rng1 = np.random.default_rng(seed=SEED) # to ensure that we get the same shuffling everytime as long as the seed stays the same, change the seed and you reshuffle the indices
    index = np.arange(0,number_of_data_values) # prints a range from 0 to end-1, i.e., the input is the number of numbers to print starting from zero and in steps of one by default, need to specify the start to avoid multiple numbers in split arrays
    rng1.shuffle(index) # shuffles the order of elements in the specified array
    groups= np.array_split(index,number_of_groups) # splits the indices into a specified number of individual arrays 
    test_indices = index[np.isin(index,groups[test_group])] # take the first array from the splits and find the indicies in that array in the group of overall indices, this creates a boolean index with the positions where the matching indices are as true, this is used to index all of the indices and create a test index array - these are the images that will be used for testing
    train_indices = index[np.logical_not(np.isin(index, test_indices))] # find the test indices in index and mark those positions with a boolean true, turn these values into boolean false, use this to index all the indices and get everything that is not a test index
    return train_indices, test_indices

def human_reliability(data, runs = 20):
    e_array = np.empty(runs)
    for x in range(runs):
        np.random.seed(x)
        random_indices = np.random.choice(data.shape[1], size=data.shape[1], replace=False)
        m_1 = np.nanmean(data[:,random_indices[0:(int((data.shape[1])/2))]], axis = 1)
        m_2 = np.nanmean(data[:,random_indices[int(((data.shape[1])/2)):int((data.shape[1]))]], axis = 1)
        r = pearsonr(m_1,m_2)[0]
        e_array[x] = 2 * r / (1 + r) # applying spearman brown corrrelation splithalf reliability correction 
    m_n = np.mean(e_array)
    s_n = np.std(e_array)
    return e_array,m_n, s_n
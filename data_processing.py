import pandas as pd
import numpy as np

def get_training_and_test_data(filepath='train_sortiert_new_format.csv', training_split=0.9):
    """
    Loads the data, converts it to a np.arr, and generates the test and train split.
    Returns: x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(filepath)
    indices = df.index.tolist()
    features = df.columns.tolist()
    global n_frames, n_landmarks, n_dimensions, n_samples
    # Set constants
    n_frames = 37
    n_landmarks = 21
    n_dimensions = 3
    n_samples = len(indices)
    n_features = len(features)
    print(f"{n_samples} samples with {n_features} features.")
    print(f"video_id + label_id + label + {n_frames} Frames * {n_landmarks} Landmarks * {n_dimensions} Dimensions -> {3+n_frames*n_landmarks*n_dimensions}")

    data = np.zeros((n_samples, n_frames, n_landmarks, n_dimensions), dtype=np.float32)
    dim_dict = {0: "X", 1: "Y", 2: "Z"}

    # Loop through all labels and store in data array
    for f in range(1,n_frames+1):
        for l in range(n_landmarks):
            for d in range(n_dimensions):
                index = "F"+str(f)+"_L"+str(l)+"_"+dim_dict[d]
                data[:,f-1,l,d] = np.array(df[index])

    print("The shape of the data is:      ", data.shape) # (n_samples, n_frames, n_landmarks, n_dimensions)
    print("The percentage of 0 entries is:", np.round(np.sum(data<=0) / np.size(data)*100,3),"%")
    print("The array has a size of:       ", np.round(data.nbytes/1024**2,2),"MB.")

    # random permutation of the indices
    np.random.seed(42)                                      # Fix seed for reproducibility
    indices = np.random.permutation(len(data))
    split_index = int(len(data) * (1 - training_split))

    # generate split index & split the data
    train_indices = indices[split_index:]
    test_indices = indices[:split_index]
    labels = np.array(df["label_id"])
    x_train, x_test = data[train_indices], data[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    # x_train_flat = x_train.reshape(x_train.shape[0], -1)
    # x_test_flat = x_test.reshape(x_test.shape[0], -1)

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Labels: ", np.unique(labels))
    return x_train, x_test, y_train, y_test

def remove_sparse_samples(arr1, arr2, sparsity): 
    """
    Removes samples that contain too many empty frames in arr1.
    Also removes the indices in arr2. "sparsity" specifies the allowed sparsity.
    """
    n = np.sum(arr1==0, axis=1) / arr1.shape[1]
    ind = n <= sparsity
    return arr1[ind], arr2[ind]

def return_single_frame(x, y, sparsity, flat=True):
    """
    Creates new samples, each containing only one frame.
    """
    # print("old x: ",x.shape)
    # print("old y: ",y.shape)

    # Convert dataset to single frames
    new_x = x.reshape(x.shape[0]*x.shape[1],-1)
    new_y = np.repeat(y, x.shape[1])
    
    new_x, new_y = remove_sparse_samples(new_x, new_y, sparsity) # remove sparse samples

    if not flat:  # Reshape if you dont want to get the flat array
        new_x = new_x.reshape(-1,x.shape[2],x.shape[3])
        new_y = new_y.reshape(-1)

    print("new_x: ",new_x.shape)
    print("new_y: ",new_y.shape)
    return new_x, new_y

def return_multi_frame(x, y, n_frames, sparsity, flat=True):
    """
    Creates new samples with specified frames.
    For example if n_frames is one smaller than the original n_frames, then you will get double the samples.
    """
    # print("old x: ",x.shape)
    # print("old y: ",y.shape)

    # 1. Step Convolve and create new samples
    new_x = np.zeros((x.shape[0], x.shape[1]-n_frames+1, n_frames, x.shape[2], x.shape[3]), dtype=np.float32)
    new_y = np.zeros((y.shape[0], x.shape[1]-n_frames+1))

    for j, sample in enumerate(x):
        for i in range(x.shape[1]-n_frames+1):
            new_x[j,i,:,:,:] = sample[i:i+n_frames,:,:]
            new_y[j,i] = y[j]

    # 2. Step Filter samples with high sparsity
    new_x = new_x.reshape(new_x.shape[0]*new_x.shape[1], -1)   # First flatten to remove sparse samples
    new_y = new_y.reshape(new_y.shape[0]*new_y.shape[1])
    new_x, new_y = remove_sparse_samples(new_x, new_y, sparsity) # remove sparse samples

    if not flat:  # Reshape if you dont want to get the flat array
        new_x = new_x.reshape(-1,n_frames,x.shape[2],x.shape[3])
        new_y = new_y.reshape(-1)

    print("new_x: ",new_x.shape)
    print("new_y: ",new_y.shape)
    return new_x, new_y

def cal_hand_metric(sample):
    """
    Returns the 21 euclidian distances between the 21 landmarks averaged over all frames of given sample
    Input:  (n_frames, n_landmarks, n_dimensions)-shaped arr
    Output: (n_landmark_connection_lengths,)-shaped arr
    """
    connections = [(0,1), (1,2), (2,3), (3,4), 
                   (0,5), (5,6), (6,7), (7,8),
                   (5,9), (9,10), (10,11), (11,12),
                   (9,13), (13,14), (14,15), (15,16),
                   (13,17), (17,18), (18,19), (19,20),
                   (0,17)]

    hand_metric = []
    for connection in connections:
        a = sample[:,connection[0],:]
        b = sample[:,connection[1],:]
        length = np.sqrt(np.sum(np.square(a-b), axis=1))
        hand_metric.append(np.mean(length))

    return hand_metric # (21,)

def augment_with_finger_length(x, y, n, alpha):
    """Augments the data array with new samples that vary in hand dimensions from the original one.

    Inputs: 
    x:          (sample, n_frames, n_landmarks, n_dimensions)-shaped arr containing the feature data
    y:          (sample,)-shaped arr containing the original labels
    n:          int - how many new samples for one orginial one?
    alpha:      percent - how much should the new fingerlength vary?

    Outputs:
    x_aug:      (sample, n_frames, n_landmarks, n_dimensions)-shaped arr containing the augmented feature data
    y_aug:          (sample,)-shaped arr containing the augmented labels
    """
    
    x_aug, y_aug = [], []

    for i, sample in enumerate(x):
        x_aug.append(sample)    # Append original sample
        y_aug.append(y[i])      # Append original label
        if i%2000==0: print(round(i/len(x)*100,2),"%")   # Print progress
        
        # Calculate hand metrics
        hand_metric = cal_hand_metric(sample)   # (21,)

        for _ in range(n):
            # Create new hand metrics with Gaussian noise
            new_hand_metric = np.array(hand_metric) * np.random.normal(1, alpha / 100.0, len(hand_metric)) # (21,)
            
            # Initialize new sample as a copy of the original sample
            new_sample = np.copy(sample)

            # Adjust landmarks in the new sample to match the new hand metrics
            connections = [(0,1), (1,2), (2,3), (3,4), 
                           (0,5), (5,6), (6,7), (7,8),
                           (5,9), (9,10), (10,11), (11,12),
                           (9,13), (13,14), (14,15), (15,16),
                           (13,17), (17,18), (18,19), (19,20),
                           (0,17)]
            for idx, connection in enumerate(connections):
                a_idx, b_idx = connection
                original_length = hand_metric[idx]
                new_length = new_hand_metric[idx]

                scale_factor = new_length / original_length if original_length != 0 else 1.0

                direction = new_sample[:, b_idx, :] - new_sample[:, a_idx, :]
                new_sample[:, b_idx, :] = new_sample[:, a_idx, :] + direction * scale_factor
            
            # print(np.max(new_sample))
            x_aug.append(new_sample) # Append new sample
            y_aug.append(y[i])       # Append label for the new sample

    return np.array(x_aug, dtype=np.float32), np.array(y_aug, dtype=np.float32)

def normalize(x):
    return np.nan_to_num((x - np.mean(x, axis=(1, 2), keepdims=True)) / np.std(x, axis=(1, 2), keepdims=True), nan=0)
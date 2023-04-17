import numpy as np
import pickle


with open("dataset/mnist/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", 'rb') as f:
    df_arr = f.readlines()
df = []
for dpt in df_arr:
    df += dpt
magic_number = int.from_bytes(df[:4], 'big')
num_images = int.from_bytes(df[4:8], 'big')
rows = int.from_bytes(df[8:12], 'big')
cols = int.from_bytes(df[12:16], 'big')
pixel_array = []
for i in range(16, len(df)):
    pixel_array.append(int.from_bytes(df[i:i+1], 'big'))
pixel_array = np.asarray(pixel_array, dtype=np.uint8)
dataset = pixel_array.reshape((num_images, rows, cols))


with open("dataset/mnist/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", 'rb') as f:
    df_arr = f.readlines()
df = []
for dpt in df_arr:
    df += dpt
magic_number = int.from_bytes(df[:4], 'big')
num_items = int.from_bytes(df[4:8], 'big')
pixel_array = []
for i in range(8, len(df)):
    pixel_array.append(int.from_bytes(df[i:i+1], 'big'))
labels_array = np.asarray(pixel_array, dtype=np.uint8)

training = {
    'images': dataset,
    'labels': labels_array
}

with open("dataset/mnist/test_df.pkl", "wb") as f:
    pickle.dump(training, f)


import pyarrow as pa

# print("Reading arrow file...")
# mmap = pa.memory_map('C:/Users/dhlabadmin/Desktop/m-test/party/dataset.arrow')

# with mmap as source:
#     array = pa.ipc.open_file(source).read_all()

# print(array[0])
# print(array[1])
# print(array.schema)

import pyarrow.ipc as ipc
import numpy as np

dataset_path = 'C:/Users/dhlabadmin/Desktop/m-test/party/dataset.arrow'
with pa.memory_map(dataset_path, 'rb') as source:
    table = ipc.open_file(source).read_all()

num_rows = table.num_rows
ids = np.random.permutation(num_rows)
split = int(0.8 * num_rows)

train_ids = ids[:split]
val_ids = ids[split:]

train_table = table.take(train_ids)
val_table = table.take(val_ids)

train_path = 'train.arrow'
val_path = 'val.arrow'

with pa.OSFile(train_path, 'wb') as sink:
    with ipc.new_file(sink, train_table.schema) as writer:
        writer.write(train_table)

with pa.OSFile(val_path, 'wb') as sink:
    with ipc.new_file(sink, val_table.schema) as writer:
        writer.write(val_table)

with open('train.lst', 'w', encoding="utf-8") as f:
    f.write(train_path + '\n')

with open('val.lst', 'w', encoding="utf-8") as f:
    f.write(val_path + '\n')

print('Done')

# file = open('train.lst', encoding='utf8')

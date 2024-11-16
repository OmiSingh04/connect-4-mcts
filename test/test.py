import h5py
import json
class QFunction:
    def __init__(self):
        self.qtable = {}

    def save_dict_to_hdf5(self, file_path):
        with h5py.File(file_path, 'w') as hdf5_file:
            for key, inner_dict in self.qtable.items():
                str_key = json.dumps(key)  # Serialize tuple key as string
                group = hdf5_file.create_group(str_key)
                for inner_key, value in inner_dict.items():
                    group.create_dataset(str(inner_key), data=value)

    def load_dict_from_hdf5(self, file_path):
        data_dict = {}
        with h5py.File(file_path, 'r') as hdf5_file:
            for str_key in hdf5_file.keys():
                tuple_key = tuple(json.loads(str_key))  # Deserialize string back to tuple
                inner_dict = {int(k): float(v[()]) for k, v in hdf5_file[str_key].items()}
                data_dict[tuple_key] = inner_dict

        self.qtable = data_dict


qf = QFunction()

print(qf.qtable)
qf.qtable = {
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 114, 15, 16, 17, 18) : {1 : 0.55, 2 : 0.98}, 
    (1, 2, 3, 4, 5, 6, 7, 8, 100, 10, 11, 12, 13, 114, 15, 16, 17, 18) : {1 : 0.55, 2 : 0.98}
}

#qf.save_dict_to_hdf5("test.hdf5")
qf.load_dict_from_hdf5("test.hdf5")
print(qf.qtable)
print(qf.qtable[(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 114, 15, 16, 17, 18)])

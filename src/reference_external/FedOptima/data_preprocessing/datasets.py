import os
import pickle


def IMDB(root, split='train'):
    assert split in ['train', 'test']
    data_file = os.path.join(root, 'IMDB_' + split + '.pickle')
    if os.path.isfile(data_file):
        with open(data_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    dataset = []
    data_folder = os.path.join(root, split)
    label_index = {
        'pos': 1,
        'neg': 0
    }
    for label in ['pos', 'neg']:
        label_folder = os.path.join(data_folder, label)
        for filename in os.listdir(label_folder):
            filepath = os.path.join(label_folder, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    dataset.append((f.readline(), label_index[label]))
    with open(data_file, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


def SST2(root, split='train'):
    assert split in ['train', 'test', 'dev']
    data_file = os.path.join(root, 'SST2_' + split + '.pickle')
    if os.path.isfile(data_file):
        with open(data_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    dataset = []
    file_name = os.path.join(root, split, '{}.tsv'.format(split))
    with open(file_name, 'r') as f:
        for line in f.readlines()[1:]:
            text, label = line.split('\t')
            dataset.append((text.strip(), int(label.strip())))
    with open(data_file, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset

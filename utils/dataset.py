from myargs import args
from torch.utils import data
from utils.utils import find_file
import torch


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, datapath, set_to_use):
        """
        Initializes dataset for given datapath
        :param datapath: path to data
        :param set_to_use: which dataset to use in dataloader
        """

        # get paths
        if set_to_use == 'train':
            datapaths = find_file(datapath + '/train', '.file_extension')

        elif set_to_use == 'val':
            datapaths = find_file(datapath + '/val', '.file_extension')

        elif set_to_use == 'test':
            datapaths = find_file(datapath + '/test', '.file_extension')

        else:
            raise ValueError('invalid type of dataset to use, must be train, val, or test')

        # true if using validation or test set, otherwise train set so not evaluating
        self.eval = True if set_to_use == 'val' or set_to_use == 'test' else False

        # build datalist
        self.datalist = []
        for path in datapaths:
            # load data here and append it
            self.datalist.append({'data': None, 'label': None})

    def __len__(self):
        """
        Denotes the total number of samples
        :return: length of the dataset
        """

        return len(self.datalist)

    def get_aug_and_tensor(self, x):
        """
        Augments data and converts it to a tensor with the proper dimensions for model
        :param x: the data to be converted to tensor and normalized
        :return: the FloatTensor format of the data
        """

        # augment here if needed
        if not self.eval:
            print('apply augmentation here!')

        # transform to tensor and normalize here
        x = torch.from_numpy(x)

        return x.float()

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index: index of the data in the datalist
        :return: returns the data / labels in float / longtensor format
        """

        data = self.datalist[index]['data']
        label = self.datalist[index]['label']

        return data, label


def generate_iterator(datapath, set_to_use='train', shuffle=True):
    """
    Generates a batch iterator for data
    :param datapath: path to data
    :param set_to_use: which dataset to use in dataloader
    :param shuffle: whether to shuffle the batches around or not
    :return: a iterator combining the data into batches
    """

    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'pin_memory': False,
        'drop_last': False,
        'num_workers': args.workers
    }

    return data.DataLoader(Dataset(datapath=datapath, set_to_use=set_to_use), **params)

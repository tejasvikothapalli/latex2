import os
import numpy as np
import pandas as pd
import torch
import random
#from skimage import io, transform
from torch.utils.data import Dataset, Sampler

SEQ_LEN = 150

class StringNumerizer():
    """
    Used for converting list of string tokens to numbers and vice versa
    """
    def __init__(self, vocab_file):
        with open(vocab_file) as f:
            vocabulary = f.read().split('\n')
        vocabulary = ['<PAD>', '<START>', '<END>', '<UNK>'] + vocabulary
        self.idx2sym = {i: w for i, w in enumerate(vocabulary)}
        self.sym2idx = {w: i for i, w in enumerate(vocabulary)}
        self.pad_idx = self.sym2idx['<PAD>']
        self.start_idx = self.sym2idx['<START>']
        self.end_idx = self.sym2idx['<END>']
        self.unk_idx = self.sym2idx['<UNK>']

    def seq2number(self, tokenized):
        """
        Converts TOKENIZED string sequence to numbers
        """
        return [self.sym2idx.get(token, self.unk_idx) for token in tokenized]

    def number2seq(self, numerized):
        """
        Converts NUMERIZED sequence back to tokenized string sequence
        """
        return [self.idx2sym.get(num, '<UNK>') for num in numerized]

    def pad_sequence(self, numerized, to_length):
        """
        Adds padding to specified NUMERIZED sequence
        Returns new numerized sequence AND a mask corresponding to the sequence
        """
        pad = numerized[:to_length]
        padded = pad + [self.pad_idx] * (to_length - len(pad))
        mask = [w != self.pad_idx for w in padded]
        return padded, mask

    def start_end_sequence(self, numerized):
        return [self.start_idx] + numerized + [self.end_idx]
        

class Image2LatexDataset(Dataset):
    """
    Dataset of Latex-image pairs
    """
    def __init__(self, data_file, image_folder, formula_list, string_numerizer):
        self.data = pd.read_csv(data_file, 
                                sep=' ', 
                                header=None, 
                                names=['img_name', 'formula_line_num'])
        self.formula_list = formula_list
        self.image_folder = image_folder
        self.string_numerizer = string_numerizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary of:
            {
                'image': in numpy format,
                'formula': latex ground truth as string
                'numerized': padded array of latex ground truth, where string is numerized
                'mask': for calculating the loss, pad should not be part of the loss calculation
            }
        """
        img_name, formula_idx = self.data.iloc[idx]

        image = io.imread(os.path.join(self.image_folder, img_name))
        image = image.transpose((2, 0, 1))

        formula = self.formula_list[formula_idx]
        numerized = self.string_numerizer.seq2number(formula.split(' '))
        numerized = self.string_numerizer.start_end_sequence(numerized)
        padded, mask = self.string_numerizer.pad_sequence(numerized, SEQ_LEN + 1)
        return {'image': image, 
                'formula': formula,
                'numerized': padded,
                'mask': mask}


class ImageBatchSampler(Sampler):
    """
    Batches images that have the same size together (lazily)
    Achieves this using a dictionary of queues for each image size.
    Once a queue reaches the desired batch size, it is flushed, and
    the contents of the queue are yielded.

    Do note that __len__ in this case would be an approximation (lower bound)
    of the number of batches will be returned
    """
    def __init__(self, dataset, batch_size, shuffle=False):
        self.image_queues = {}
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        for idx in self.indices:
            image_shape = self.dataset[idx]['image'].shape
            if image_shape not in self.image_queues:
                self.image_queues[image_shape] = []
            queue = self.image_queues[image_shape]
            queue.append(idx)
            if len(queue) >= self.batch_size:
                self.image_queues[image_shape] = []
                yield queue

        for queue in self.image_queues.values():
            if len(queue) > 0:
                yield queue

    def __len__(self):
        return len(self.indices) // batch_size
        

def image2latex_collate_fn(batch):
    img_list = []
    formula_list = []
    numerized_list = []
    mask_list = []

    for b in batch:
        img_list.append(b['image'])
        formula_list.append(b['formula'])
        numerized_list.append(b['numerized'])
        mask_list.append(b['mask'])
    
    return {'image': torch.FloatTensor(img_list),
            'formula': formula_list,
            'numerized': torch.LongTensor(numerized_list),
            'mask': torch.FloatTensor(mask_list)}

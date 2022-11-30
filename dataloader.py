import os
import ast
import utils.binvox_rw
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class CadDataset(Dataset):
    

    def __init__(self, config, data_path, training=True):

        # Getting the Training/Testing Data.
        data = pd.read_csv(data_path)

        self.files_name = [i for i in data['0']]
        self.captions = [i for i in data['encoded_labels']]

        self._inference_captions = self._group_captions(self.files_name, self.captions)

        self._training = training

        # Auxiliary token indices
        self._start_idx = config["START_idx"]
        self._end_idx = config["END_idx"]
        self._pad_idx = config["PAD_idx"]

        self._max_len = config["max_len"]

    def _group_captions(self, files_name, captions):
        """Groups captions which correspond to the same image.
        Main usage: Calculating BLEU score
        Arguments:
            data (list of str): Each element contains image name and corresponding caption
        Returns:
            grouped_captions (dict): Key - image name, Value - list of captions associated
                with that picture
        """
        grouped_captions = {}

        for i, value in enumerate(files_name):
            if value not in grouped_captions:
                # We came across the first caption for this particular image
                grouped_captions[value] = []

            grouped_captions[value].append(captions[i])

        return grouped_captions

    def inference_batch(self, batch_size):
        """Creates a mini batch dataloader for inference.
        During inference we generate caption from scratch and in each iteration
        we feed words generated previously by the model (i.e. no teacher forcing).
        We only need input image as well as the target caption.
        """
        caption_data_items = list(self._inference_captions.items())
        # random.shuffle(caption_data_items)

        num_batches = len(caption_data_items) // batch_size
        for idx in range(num_batches):
            caption_samples = caption_data_items[idx * batch_size: (idx + 1) * batch_size]
            batch_imgs = []
            batch_captions = []

            # Increase index for the next batch
            idx += batch_size

            # Create a mini batch data
            for image_name, captions in caption_samples:
                batch_captions.append(captions)
                batch_imgs.append(torch.tensor(self._load_and_prepare_image(image_name)))

            # Batch image tensors
            batch_imgs = torch.stack(batch_imgs, dim=0)
            if batch_size == 1:
                batch_imgs = batch_imgs.unsqueeze(0)

            yield batch_imgs, batch_captions

    def _load_and_prepare_image(self, image_name):
        
        image_path = os.path.join('./Voxels_transformer/Voxels', image_name)

        with open(image_path,'rb') as f:
            voxel_data = utils.binvox_rw.read_as_3d_array(f)
            voxel_data = voxel_data.data.astype(np.float32)

        image_tensor = voxel_data
        return image_tensor

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, index):
        
        # Extract the caption data
        image_id = self.files_name[index]
        tokens = self.captions[index]

        # Load and preprocess image
        image_tensor = self._load_and_prepare_image(image_id)

        # Pad the token and label sequences
        tokens = ast.literal_eval(tokens)
        tokens = tokens[:self._max_len]

        # Extract input and target output
        input_tokens = tokens[:-1].copy()
        tgt_tokens = tokens[1:].copy()

        # Number of words in the input token
        sample_size = len(input_tokens)
        padding_size = self._max_len - sample_size

        # Add direclty <pad> = 99 in the input_tokens
        if padding_size > 0:
            padding_vec = [self._pad_idx for _ in range(padding_size)]
            input_tokens += padding_vec.copy()
            tgt_tokens += padding_vec.copy()

        input_tokens = torch.Tensor(input_tokens).long()
        tgt_tokens = torch.Tensor(tgt_tokens).long()

        # Index from which to extract the model prediction
        # Define the padding masks
        tgt_padding_mask = torch.ones([self._max_len, ])
        tgt_padding_mask[:sample_size] = 0.0
        tgt_padding_mask = tgt_padding_mask.bool()

        return image_tensor, input_tokens, tgt_tokens, tgt_padding_mask

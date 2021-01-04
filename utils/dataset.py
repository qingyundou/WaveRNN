import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
from utils import hparams as hp
from utils.text import text_to_sequence
from utils.paths import Paths
from pathlib import Path

import warnings


###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class VocoderDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, train_gta=False, voc_model_id=''):
        self.metadata = dataset_ids
        self.mel_path = path/'gta' if train_gta else path/'mel'
        if train_gta and (voc_model_id != ''):
            self.mel_path = path/f'gta_{voc_model_id}' # train NV for a sepcific Taco
            print(f'using conditioning vectors at {self.mel_path}')
        self.quant_path = path/'quant'


    def __getitem__(self, index):
        item_id = self.metadata[index]
        m = np.load(self.mel_path/f'{item_id}.npy')
        x = np.load(self.quant_path/f'{item_id}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path: Path, batch_size, train_gta, voc_model_id=''):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(path, train_ids, train_gta, voc_model_id)
    test_dataset = VocoderDataset(path, test_ids, train_gta, voc_model_id)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels


###################################################################################
# Tacotron/TTS Dataset ############################################################
###################################################################################


def get_tts_datasets(path: Path, batch_size, r, data_split=[-1,0,0]):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    mel_lengths = []

    # get lst of valid and test data
    v_t_id_lst = get_v_t_id_lst(dataset, data_split)

    # filter out valid and test data, and long data
    for (item_id, len) in dataset:
        if (len <= hp.tts_max_mel_len) and (item_id not in v_t_id_lst): 
            dataset_ids += [item_id]
            mel_lengths += [len]

    with open(path/'text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f)

    train_dataset = TTSDataset(path, dataset_ids, text_dict)

    sampler = None

    if hp.tts_bin_lengths:
        sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=1,
                           pin_memory=True)

    longest = mel_lengths.index(max(mel_lengths))

    # Used to evaluate attention during training process
    attn_example = dataset_ids[longest]

    # print(attn_example)

    return train_set, attn_example

def get_v_t_id_lst(dataset, data_split):
    # clean data_split
    if -1 in data_split:
        assert data_split.count(-1)==1, f'expected at most one -1 in data_split but got {data_split.count(-1)}'
        tmp = len(dataset) - (sum(data_split)+1)
        data_split = [tmp if x==-1 else x for x in data_split]
    assert len(dataset)==sum(data_split), f'len(dataset) {len(dataset)} != sum(data_split) {sum(data_split)}'

    #  using safe indexing; if there is a split, sort id_lst and get v_t_id_lst
    if len(dataset)==data_split[0]:
        return []
    else:
        v_t_id_lst = sorted([item_id for item_id,l in dataset])
        return v_t_id_lst[-data_split[1]-data_split[2]:]


class TTSDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, text_dict):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        item_id = self.metadata[index]
        x = text_to_sequence(self.text_dict[item_id], hp.tts_cleaner_names)
        mel = np.load(self.path/'mel'/f'{item_id}.npy')
        mel_len = mel.shape[-1]
        if hp.mode in ['teacher_forcing', 'attention_forcing_online']:
            return x, mel, item_id, mel_len
        elif hp.mode == 'attention_forcing_offline':
            attn_ref = np.load(self.path/hp.attn_ref_path/f'{item_id}.npy')
            return x, mel, item_id, mel_len, attn_ref

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def pad_cut_attn(attn, max_x_len, max_attn_len):
    l_a, l_x = attn.shape
    attn_pad = attn

    if max_x_len - l_x < 0:
        if max_x_len < 0.5*l_x: warnings.warn(f'max_x_len {max_x_len} < 0.5 * l_x {l_x}')
        # print(max_x_len, attn.shape)
        tmp = attn_pad[:, -(1+l_x-max_x_len):-1].sum(axis=1, keepdims=True) / max_x_len
        attn_pad = np.delete(attn, np.s_[-(1+l_x-max_x_len):-1], axis = 1)
        attn_pad += tmp
    elif max_x_len - l_x > 0:
        tmp = np.zeros([max_x_len - l_x, 1])
        attn_pad = np.insert(attn,-1,tmp, axis = 1)
        
    if max_attn_len - l_a < 0:
        if max_attn_len < 0.5*l_a: warnings.warn(f'max_attn_len {max_attn_len} < 0.5 * l_a {l_a}')
        attn_pad = attn_pad[:max_attn_len]
    elif max_attn_len - l_a > 0:
        tmp = np.tile(attn_pad[-1,:], (max_attn_len - l_a,1))
        attn_pad = np.concatenate([attn_pad, tmp], axis = 0)

    return attn_pad


def collate_tts(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.

    if hp.mode in ['teacher_forcing', 'attention_forcing_online']:
        return chars, mel, ids, mel_lens
    elif hp.mode == 'attention_forcing_offline':
        # raise NotImplementedError(f'hp.mode={hp.mode} is not yet implemented')
        attn_ref = [pad_cut_attn(x[4], max_x_len, max_spec_len//r) for x in batch]
        attn_ref = np.stack(attn_ref)
        attn_ref = torch.tensor(attn_ref)
        return chars, mel, ids, mel_lens, attn_ref

    # # better code with *
    # rets = (chars, mel, ids, mel_lens)
    # if hp.mode in ['teacher_forcing', 'attention_forcing_online']:
    #     pass
    # elif hp.mode == 'attention_forcing_offline':
    #     attn_ref = [pad_cut_attn(x[4], max_x_len, max_spec_len//r) for x in batch]
    #     attn_ref = np.stack(attn_ref)
    #     attn_ref = torch.tensor(attn_ref)
    #     rets += (attn_ref,)

    # if hp.tts_use_guided_attn_loss:
    #     x_lens = torch.tensor(x_lens).long()
    #     y_lens = [l//r for l in mel_lens]
    #     y_lens = torch.tensor(y_lens).long()
    #     rets += (x_lens, y_lens)
    # return rets


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union

import sys
sys.path.append("/home/dawna/tts/qd212/models/espnet")
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as Encoder_vc_taco2_espnet


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class Encoder(nn.Module):
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        return x


class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        # Fix the highway input if necessary
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x):
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x) # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways: x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]

class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class Attention(nn.Module):
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):

        # print(encoder_seq_proj.shape)
        # Transform the query vector
        query_proj = self.W(query).unsqueeze(1)

        # Compute the scores
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)

        return scores.transpose(1, 2)


class LSA(nn.Module):
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj):
        device = next(self.parameters()).device  # use same device as parameters
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t, device=device)
        self.attention = torch.zeros(b, t, device=device)

    def forward(self, encoder_seq_proj, query, t):

        if t == 0: self.init_attention(encoder_seq_proj)

        processed_query = self.W(query).unsqueeze(1)

        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)

        # Smooth Attention
        scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True)
        # scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative += self.attention

        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(nn.Module):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims):
        super().__init__()
        self.register_buffer('r', torch.tensor(1, dtype=torch.int))
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels)
        self.attn_net = LSA(decoder_dims)
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)

    def zoneout(self, prev, current, p=0.1):
        device = next(self.parameters()).device  # Use same device as parameters
        mask = torch.zeros(prev.size(), device=device).bernoulli_(p)
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, t, attn_ref=None):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        # import pdb; pdb.set_trace()
        prenet_out = self.prenet(prenet_in)

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)

        # Dot product to create the context vector
        if attn_ref is None:
            context_vec = scores @ encoder_seq
        else:
            # import pdb; pdb.set_trace()
            # print(attn_ref.size())
            # print(scores.size())
            # pdb.set_trace()
            context_vec = attn_ref @ encoder_seq # attention forcing
        context_vec = context_vec.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, hidden_states, cell_states, context_vec


class Tacotron(nn.Module):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing'):
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, decoder_dims, lstm_dims)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [256, 80], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, fft_bins, bias=False)

        self.init_model()
        self.num_params()

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('stop_threshold', torch.tensor(stop_threshold, dtype=torch.float32))

        self.mode = mode

    @property
    def r(self):
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, x, m, generate_gta=False, attn_ref=None):
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores

    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
        elif self.mode=='free_running':
            device = next(self.parameters()).device
            # mask = torch.ones([m.size(0)], device=device)

            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)

                # mask = mask * (mel_frames >= self.stop_threshold).int().float().prod(-1).prod(-1)
                # tmp = mel_frames * mask.unsqueeze(-1).unsqueeze(-1) + (-4) * (1-mask).unsqueeze(-1).unsqueeze(-1)
                # mel_outputs.append(tmp)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break
                # import pdb; pdb.set_trace()

        return mel_outputs, attn_scores

    def generate(self, x, steps=2000):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
            self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                         hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            # Stop the loop if silent frames present
            if (mel_frames < self.stop_threshold).all() and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)


        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]

        self.train()

        return mel_outputs, linear, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)

        # Backwards compatibility with old saved models
        if 'r' in state_dict and not 'decoder.r' in state_dict:
            self.r = state_dict['r']

        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters


# --------------------------------
class Tacotron_ss(Tacotron):
    def forward(self, x, m, generate_gta=False, attn_ref=None, p_tf=1.):
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=attn_ref, p_tf=p_tf)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores

    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=None, p_tf=1.):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)

        if self.mode=='scheduled_sampling':
            for t in range(0, steps, self.r):
                # prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                if t==0:
                    prenet_in = go_frame
                elif np.random.uniform(high=1.0) <= p_tf:
                    prenet_in = m[:, :, t - 1]
                else:
                    prenet_in = mel_outputs[-1][:, :, -1]
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)

        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)

        elif self.mode=='free_running':
            device = next(self.parameters()).device
            # mask = torch.ones([m.size(0)], device=device)

            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break
                # import pdb; pdb.set_trace()

        return mel_outputs, attn_scores


# --------------------------------
class Tacotron_pass1(Tacotron):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 fr_length_ratio=1, share_encoder=False):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode=mode)
        self.fr_length_ratio = fr_length_ratio
        self.share_encoder = share_encoder


    def forward(self, x, m, generate_gta=False, generate_fr=False, attn_ref=None):
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta or generate_fr:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores, attn_hiddens = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        # qd212, new for Taco_pass1, Concat the mel outputs into sequence
        attn_hiddens = torch.cat(attn_hiddens, dim=2)

        # print(len(mel_outputs), mel_outputs[0].size())
        # print(len(attn_hiddens), attn_hiddens[0].size())
        # import pdb; pdb.set_trace()

        ret = (mel_outputs, linear, attn_scores, attn_hiddens)
        if self.share_encoder: ret += (encoder_seq, encoder_seq_proj)
        return ret

    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_hiddens = [], [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_hiddens.append(hidden_states[0].unsqueeze(-1))
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_hiddens.append(hidden_states[0].unsqueeze(-1))
        elif self.mode=='free_running':
            device = next(self.parameters()).device
            # mask = torch.ones([m.size(0)], device=device)
            # mask = torch.ones([encoder_seq.size(0)], device=device)

            for t in range(0, int(steps*self.fr_length_ratio), self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)

                # mask = mask * (mel_frames >= self.stop_threshold).int().float().prod(-1).prod(-1)
                # tmp = mel_frames * mask.unsqueeze(-1).unsqueeze(-1) + (-4) * (1-mask).unsqueeze(-1).unsqueeze(-1)
                # mel_outputs.append(tmp)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_hiddens.append(hidden_states[0].unsqueeze(-1))
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break
                # import pdb; pdb.set_trace()

        return mel_outputs, attn_scores, attn_hiddens

    def generate(self, x, steps=2000, m=None, attn_ref=None):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        if self.mode!='free_running':
            import warnings
            msg = f'the mode is not free_running but {self.mode}'
            print(msg)
            warnings.warn(msg)
        mel_outputs, attn_scores, attn_hiddens = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #     self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                  hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)
        #     # Stop the loop if silent frames present
        #     if (mel_frames < self.stop_threshold).all() and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)


        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]

        # qd212, new for Taco_pass1, Concat the mel outputs into sequence
        attn_hiddens = torch.cat(attn_hiddens, dim=2)
        # no need to put to cpu, as this will be used by Taco_pass2, unless the code runs on cpu

        self.train()

        ret = (mel_outputs, linear, attn_scores, attn_hiddens)
        if self.share_encoder: ret += (encoder_seq, encoder_seq_proj)
        return ret

# --------------------------------

class Encoder_vc(Encoder):
    def __init__(self, embed_dims, n_mels, cbhg_channels, K, num_highways, dropout):
        super().__init__(embed_dims, n_mels, cbhg_channels, K, num_highways, dropout)
        # overwrite
        self.embedding = nn.Linear(n_mels, embed_dims)


class Decoder_pass2(Decoder):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims):
        super().__init__(n_mels, decoder_dims, lstm_dims)
        # new
        self.attn_net_vc = LSA(decoder_dims)
        self.attn_rnn_vc = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input_vc = nn.Linear(2 * decoder_dims, lstm_dims)


    def forward(self, encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in, 
        hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=None, attn_ref_vc=None):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)
        attn_rnn_in_vc = torch.cat([context_vec_vc, prenet_out], dim=-1)
        attn_hidden_vc = self.attn_rnn_vc(attn_rnn_in_vc.squeeze(1), attn_hidden_vc)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        scores_vc = self.attn_net_vc(encoder_seq_proj_vc, attn_hidden_vc, t)

        # Dot product to create the context vector
        # if (attn_ref is None) or (attn_ref_vc is None):
        #     context_vec = scores @ encoder_seq
        #     context_vec_vc = scores_vc @ encoder_seq_vc
        # else:
        #     # import pdb; pdb.set_trace()
        #     # print(attn_ref.size())
        #     # print(scores.size())
        #     # pdb.set_trace()
        #     context_vec = attn_ref @ encoder_seq # attention forcing
        #     context_vec_vc = attn_ref_vc @ encoder_seq_vc
        context_vec = scores @ encoder_seq if attn_ref is None else attn_ref @ encoder_seq
        context_vec_vc = scores_vc @ encoder_seq_vc if attn_ref_vc is None else attn_ref_vc @ encoder_seq_vc

        context_vec = context_vec.squeeze(1)
        context_vec_vc = context_vec_vc.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)
        x_vc = torch.cat([context_vec_vc, attn_hidden_vc], dim=1)
        x_vc = self.rnn_input_vc(x_vc)
        x = x + x_vc

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc


class Tacotron_pass2(Tacotron):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode=mode)

        # new
        # tmp = n_mels + decoder_dims if 's1' in pass2_input else n_mels
        # self.encoder_vc = Encoder_vc(embed_dims, tmp * encoder_reduction_factor, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_vc = Encoder_vc(embed_dims, n_mels * encoder_reduction_factor, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj_vc = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.encoder_reduction_factor = encoder_reduction_factor

        self.pass2_input = pass2_input
        self.encoder_reduction_factor_s = encoder_reduction_factor_s
        # print(self.encoder_reduction_factor_s)
        # import pdb; pdb.set_trace()
        if 's1' in self.pass2_input:
            # self.encoder_s = nn.Linear(decoder_dims * encoder_reduction_factor_s, n_mels * encoder_reduction_factor, bias=False)
            # overwrite
            tmp = n_mels * encoder_reduction_factor + decoder_dims * encoder_reduction_factor_s
            self.encoder_vc = Encoder_vc(embed_dims, tmp, encoder_dims, encoder_K, num_highways, dropout)

        # overwrite
        self.decoder = Decoder_pass2(n_mels, decoder_dims, lstm_dims)

        if init_model:
            self.init_model()
            self.num_params()

    def forward(self, x, m, m_p1, s_p1=None, generate_gta=False, attn_ref=None):
        """
        input
        x [B, Tin]: input text
        m [B, D, Tout]: ref output
        m_p1 [B, D, Tout']: output from the 1st pass
        attn_ref [B, Tout / r, Tin]: reference attention

        output
        m_p2 [B, D, Tout']: output from the 2nd pass
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # print(m_p1.size())
        # print(self.encoder_vc.embedding.weight.data.size())
        # import pdb; pdb.set_trace()

        # thin out input frames for reduction factor
        # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)

        # if self.encoder_reduction_factor > 1:
        #     B, idim, Lmax = m_p1.shape
        #     if Lmax % self.encoder_reduction_factor != 0:
        #         m_p1 = m_p1[:, :, : -(Lmax % self.encoder_reduction_factor)]
        #     m_p1_ds = m_p1.contiguous().view(
        #         B,
        #         idim * self.encoder_reduction_factor,
        #         int(Lmax / self.encoder_reduction_factor),
        #     )
        # else:
        #     m_p1_ds = m_p1
        # encoder_seq_vc = self.encoder_vc(m_p1_ds.transpose(1,2))

        tmp = m_p1.transpose(1,2)
        if self.encoder_reduction_factor > 1:
            B, Lmax, idim = tmp.shape
            if Lmax % self.encoder_reduction_factor != 0:
                tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor), :]
            m_p1_ds = tmp.contiguous().view(
                B,
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
        else:
            m_p1_ds = tmp

        if 's1' in self.pass2_input:
            tmp = s_p1.transpose(1,2)
            if self.encoder_reduction_factor_s > 1:
                B, Lmax, idim = tmp.shape
                if Lmax % self.encoder_reduction_factor_s != 0:
                    tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor_s), :]
                s_p1_ds = tmp.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor_s),
                    idim * self.encoder_reduction_factor_s,
                )
            else:
                s_p1_ds = tmp
            # m_p1_ds += self.encoder_s(s_p1_ds)
            m_p1_ds = torch.cat([m_p1_ds, s_p1_ds], dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        mel_outputs, attn_scores, attn_scores_vc = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc = torch.cat(attn_scores_vc, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores, attn_scores_vc

    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, 
        encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_scores_vc = [], [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
        elif self.mode=='free_running':
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                             hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break

        return mel_outputs, attn_scores, attn_scores_vc


    def generate(self, x, m_p1, s_p1=None, steps=2000):
        self.mode = 'free_running'
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = m_p1.transpose(1,2)
        if self.encoder_reduction_factor > 1:
            B, Lmax, idim = tmp.shape
            if Lmax % self.encoder_reduction_factor != 0:
                tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor), :]
            m_p1_ds = tmp.contiguous().view(
                B,
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
        else:
            m_p1_ds = tmp

        if 's1' in self.pass2_input:
            tmp = s_p1.transpose(1,2)
            if self.encoder_reduction_factor_s > 1:
                B, Lmax, idim = tmp.shape
                if Lmax % self.encoder_reduction_factor_s != 0:
                    tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor_s), :]
                s_p1_ds = tmp.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor_s),
                    idim * self.encoder_reduction_factor_s,
                )
            else:
                s_p1_ds = tmp
            # m_p1_ds += self.encoder_s(s_p1_ds)
            m_p1_ds = torch.cat([m_p1_ds, s_p1_ds], dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        mel_outputs, attn_scores, attn_scores_vc = self.decoder_loop(steps, m_p1, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc = torch.cat(attn_scores_vc, 1)

        # mv to cpu
        linear = linear[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = attn_scores.cpu().data.numpy()[0]
        attn_scores_vc = attn_scores_vc.cpu().data.numpy()[0]
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores, attn_scores_vc


# --------------------------------

class Decoder_pass2_concat(Decoder):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims):
        super().__init__(n_mels, decoder_dims, lstm_dims)
        # new
        self.attn_net_vc = LSA(decoder_dims)
        self.attn_rnn_vc = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input_concat = nn.Linear(2 * 2 * decoder_dims, lstm_dims)


    def forward(self, encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in, 
        hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=None, attn_ref_vc=None):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)
        attn_rnn_in_vc = torch.cat([context_vec_vc, prenet_out], dim=-1)
        attn_hidden_vc = self.attn_rnn_vc(attn_rnn_in_vc.squeeze(1), attn_hidden_vc)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        scores_vc = self.attn_net_vc(encoder_seq_proj_vc, attn_hidden_vc, t)

        # Dot product to create the context vector
        # if (attn_ref is None) or (attn_ref_vc is None):
        #     context_vec = scores @ encoder_seq
        #     context_vec_vc = scores_vc @ encoder_seq_vc
        # else:
        #     # import pdb; pdb.set_trace()
        #     # print(attn_ref.size())
        #     # print(scores.size())
        #     # pdb.set_trace()
        #     context_vec = attn_ref @ encoder_seq # attention forcing
        #     context_vec_vc = attn_ref_vc @ encoder_seq_vc
        context_vec = scores @ encoder_seq if attn_ref is None else attn_ref @ encoder_seq
        context_vec_vc = scores_vc @ encoder_seq_vc if attn_ref_vc is None else attn_ref_vc @ encoder_seq_vc

        context_vec = context_vec.squeeze(1)
        context_vec_vc = context_vec_vc.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        # x = self.rnn_input(x)
        x_vc = torch.cat([context_vec_vc, attn_hidden_vc], dim=1)
        # x_vc = self.rnn_input_vc(x_vc)

        x = torch.cat([x, x_vc], dim=1)
        x = self.rnn_input_concat(x)

        # print(x.size(), x_vc.size())
        # import pdb; pdb.set_trace()
        # x = x + x_vc

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc


class Tacotron_pass2_concat(Tacotron_pass2):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=False)
        # overwrite
        self.decoder = Decoder_pass2_concat(n_mels, decoder_dims, lstm_dims)

        if init_model:
            self.init_model()
            self.num_params()


# --------------------------------

class Decoder_pass2_delib(Decoder):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims):
        super().__init__(n_mels, decoder_dims, lstm_dims)
        # new
        self.attn_net_vc = LSA(decoder_dims)
        # self.attn_rnn_vc = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        # self.rnn_input_concat = nn.Linear(2 * 2 * decoder_dims, lstm_dims)

        # overwrite
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input = nn.Linear(3 * decoder_dims, lstm_dims)


    def forward(self, encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in, 
        hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=None, attn_ref_vc=None):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        # attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden = hidden_states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)

        # print(prenet_out.size(), context_vec.size(), context_vec_vc.size())
        # import pdb; pdb.set_trace()

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec_vc, context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)
        # attn_rnn_in_vc = torch.cat([context_vec_vc, prenet_out], dim=-1)
        # attn_hidden_vc = self.attn_rnn_vc(attn_rnn_in_vc.squeeze(1), attn_hidden_vc)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        scores_vc = self.attn_net_vc(encoder_seq_proj_vc, attn_hidden, t)

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq if attn_ref is None else attn_ref @ encoder_seq
        context_vec_vc = scores_vc @ encoder_seq_vc if attn_ref_vc is None else attn_ref_vc @ encoder_seq_vc

        context_vec = context_vec.squeeze(1)
        context_vec_vc = context_vec_vc.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec_vc, context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)
        # x_vc = torch.cat([context_vec_vc, attn_hidden_vc], dim=1)
        # x_vc = self.rnn_input_vc(x_vc)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc


class Tacotron_pass2_delib(Tacotron):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode=mode)

        # new
        # tmp = n_mels + decoder_dims if 's1' in pass2_input else n_mels
        # self.encoder_vc = Encoder_vc(embed_dims, tmp * encoder_reduction_factor, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_vc = Encoder_vc(embed_dims, n_mels * encoder_reduction_factor, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj_vc = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.encoder_reduction_factor = encoder_reduction_factor

        self.pass2_input = pass2_input
        self.encoder_reduction_factor_s = encoder_reduction_factor_s
        # print(self.encoder_reduction_factor_s)
        # import pdb; pdb.set_trace()
        if 's1' in self.pass2_input:
            # self.encoder_s = nn.Linear(decoder_dims * encoder_reduction_factor_s, n_mels * encoder_reduction_factor, bias=False)
            # overwrite
            tmp = n_mels * encoder_reduction_factor + decoder_dims * encoder_reduction_factor_s
            self.encoder_vc = Encoder_vc(embed_dims, tmp, encoder_dims, encoder_K, num_highways, dropout)

        # overwrite
        self.decoder = Decoder_pass2_delib(n_mels, decoder_dims, lstm_dims)

        if init_model:
            self.init_model()
            self.num_params()

    def forward(self, x, m, m_p1, s_p1=None, generate_gta=False, attn_ref=None):
        """
        input
        x [B, Tin]: input text
        m [B, D, Tout]: ref output
        m_p1 [B, D, Tout']: output from the 1st pass
        attn_ref [B, Tout / r, Tin]: reference attention

        output
        m_p2 [B, D, Tout']: output from the 2nd pass
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # print(m_p1.size())
        # print(self.encoder_vc.embedding.weight.data.size())
        # import pdb; pdb.set_trace()

        # thin out input frames for reduction factor
        # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)

        # if self.encoder_reduction_factor > 1:
        #     B, idim, Lmax = m_p1.shape
        #     if Lmax % self.encoder_reduction_factor != 0:
        #         m_p1 = m_p1[:, :, : -(Lmax % self.encoder_reduction_factor)]
        #     m_p1_ds = m_p1.contiguous().view(
        #         B,
        #         idim * self.encoder_reduction_factor,
        #         int(Lmax / self.encoder_reduction_factor),
        #     )
        # else:
        #     m_p1_ds = m_p1
        # encoder_seq_vc = self.encoder_vc(m_p1_ds.transpose(1,2))

        tmp = m_p1.transpose(1,2)
        if self.encoder_reduction_factor > 1:
            B, Lmax, idim = tmp.shape
            if Lmax % self.encoder_reduction_factor != 0:
                tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor), :]
            m_p1_ds = tmp.contiguous().view(
                B,
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
        else:
            m_p1_ds = tmp

        if 's1' in self.pass2_input:
            tmp = s_p1.transpose(1,2)
            if self.encoder_reduction_factor_s > 1:
                B, Lmax, idim = tmp.shape
                if Lmax % self.encoder_reduction_factor_s != 0:
                    tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor_s), :]
                s_p1_ds = tmp.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor_s),
                    idim * self.encoder_reduction_factor_s,
                )
            else:
                s_p1_ds = tmp
            # m_p1_ds += self.encoder_s(s_p1_ds)
            m_p1_ds = torch.cat([m_p1_ds, s_p1_ds], dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        mel_outputs, attn_scores, attn_scores_vc = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc = torch.cat(attn_scores_vc, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores, attn_scores_vc

    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, 
        encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_scores_vc = [], [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
        elif self.mode=='free_running':
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                             hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break

        return mel_outputs, attn_scores, attn_scores_vc


    def generate(self, x, m_p1, s_p1=None, steps=2000):
        self.mode = 'free_running'
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = m_p1.transpose(1,2)
        if self.encoder_reduction_factor > 1:
            B, Lmax, idim = tmp.shape
            if Lmax % self.encoder_reduction_factor != 0:
                tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor), :]
            m_p1_ds = tmp.contiguous().view(
                B,
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
        else:
            m_p1_ds = tmp

        if 's1' in self.pass2_input:
            tmp = s_p1.transpose(1,2)
            if self.encoder_reduction_factor_s > 1:
                B, Lmax, idim = tmp.shape
                if Lmax % self.encoder_reduction_factor_s != 0:
                    tmp = tmp[:, : -(Lmax % self.encoder_reduction_factor_s), :]
                s_p1_ds = tmp.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor_s),
                    idim * self.encoder_reduction_factor_s,
                )
            else:
                s_p1_ds = tmp
            # m_p1_ds += self.encoder_s(s_p1_ds)
            m_p1_ds = torch.cat([m_p1_ds, s_p1_ds], dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        mel_outputs, attn_scores, attn_scores_vc = self.decoder_loop(steps, m_p1, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc = torch.cat(attn_scores_vc, 1)

        # mv to cpu
        linear = linear[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = attn_scores.cpu().data.numpy()[0]
        attn_scores_vc = attn_scores_vc.cpu().data.numpy()[0]
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores, attn_scores_vc


    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)

        # Backwards compatibility with old saved models
        if 'r' in state_dict and not 'decoder.r' in state_dict:
            self.r = state_dict['r']

        # import pdb
        # state_dict_new = self.state_dict()
        k_imcompatible_lst = []
        for k, v in state_dict.items():
            if v.size()!=self.state_dict()[k].size():
                k_imcompatible_lst.append(k)
                print(f'qd212 warning: size mismatch for {k}, old {v.size()}, new {self.state_dict()[k].size()}')
        state_dict = {k: v for k, v in state_dict.items() if k not in k_imcompatible_lst}
        # state_dict_new.update(state_dict)
        # pdb.set_trace()

        self.load_state_dict(state_dict, strict=False)


# --------------------------------

class Tacotron_pass2_delib_shareEnc(Tacotron):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True, share_encoder=True):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode=mode)

        # new
        # self.encoder_vc = Encoder_vc(embed_dims, n_mels * encoder_reduction_factor, encoder_dims, encoder_K, num_highways, dropout)
        tmp = 0
        assert 'y1' in pass2_input or 's1' in pass2_input, 'y1 & s1 are both absent from pass2_input'
        if 'y1' in pass2_input: tmp += n_mels * encoder_reduction_factor
        if 's1' in pass2_input: tmp += decoder_dims * encoder_reduction_factor_s
        self.encoder_vc = Encoder_vc(embed_dims, tmp, encoder_dims, encoder_K, num_highways, dropout)

        self.encoder_proj_vc = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.encoder_reduction_factor = encoder_reduction_factor

        self.pass2_input = pass2_input
        self.encoder_reduction_factor_s = encoder_reduction_factor_s

        # overwrite
        self.decoder = Decoder_pass2_delib(n_mels, decoder_dims, lstm_dims)
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = None
            self.encoder_proj = None

        if init_model:
            self.init_model()
            self.num_params()

    def forward(self, x, m, m_p1, s_p1=None, e_p1=None, e_p_p1=None, generate_gta=False, attn_ref=None):
        """
        input
        x [B, Tin]: input text
        m [B, D, Tout]: ref output
        m_p1 [B, D, Tout']: output from the 1st pass
        attn_ref [B, Tout / r, Tin]: reference attention

        output
        m_p2 [B, D, Tout']: output from the 2nd pass
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s))
        m_p1_ds = torch.cat(tmp, dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=attn_ref)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc = torch.cat(attn_scores_vc, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores, attn_scores_vc


    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, 
        encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_scores_vc = [], [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
        elif self.mode=='free_running':
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc, hidden_states, cell_states, context_vec, context_vec_vc = \
                self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                             hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc.append(scores_vc)
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break

        return mel_outputs, attn_scores, attn_scores_vc


    def generate(self, x, m_p1, s_p1=None, e_p1=None, e_p_p1=None, steps=2000):
        self.mode = 'free_running'
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s))
        m_p1_ds = torch.cat(tmp, dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc = self.decoder_loop(steps, m_p1, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc = torch.cat(attn_scores_vc, 1)

        # mv to cpu
        linear = linear[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = attn_scores.cpu().data.numpy()[0]
        attn_scores_vc = attn_scores_vc.cpu().data.numpy()[0]
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores, attn_scores_vc


    def get_reduced_input(self, m_p1, encoder_reduction_factor):
        tmp = m_p1.transpose(1,2)
        if encoder_reduction_factor > 1:
            B, Lmax, idim = tmp.shape
            if Lmax % encoder_reduction_factor != 0:
                tmp = tmp[:, : -(Lmax % encoder_reduction_factor), :]
            m_p1_ds = tmp.contiguous().view(
                B,
                int(Lmax / encoder_reduction_factor),
                idim * encoder_reduction_factor,
            )
        else:
            m_p1_ds = tmp
        return m_p1_ds


    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)

        # Backwards compatibility with old saved models
        if 'r' in state_dict and not 'decoder.r' in state_dict:
            self.r = state_dict['r']

        # import pdb
        state_dict_new = self.state_dict()
        k_imcompatible_lst = []
        for k, v in state_dict_new.items():
            if k in state_dict:
                if v.size()==state_dict[k].size():
                    state_dict_new[k] = state_dict[k]
                else:
                    k_imcompatible_lst.append(k)
                    print(f'qd212 warning: size mismatch for {k}, old {v.size()}, new {state_dict[k].size()}')
        # state_dict_new.update(state_dict)
        # pdb.set_trace()

        self.load_state_dict(state_dict_new, strict=False)


# --------------------------------

class Decoder_pass2_attn(Decoder):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims, nb_heads_vc=2):
        super().__init__(n_mels, decoder_dims, lstm_dims)
        # new
        self.nb_heads_vc = nb_heads_vc
        self.attn_net_vc = LSA(decoder_dims)
        self.attn_net_vc_global = Attention(decoder_dims)
        self.rnn_input_multihead = nn.Linear(2 * decoder_dims, decoder_dims)

        # overwrite
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input = nn.Linear(3 * decoder_dims, lstm_dims)


    def forward(self, encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in, 
        hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=None, attn_ref_vc=None, attn_ref_vc_global=None):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)
        # print(prenet_out.size(), context_vec.size(), context_vec_vc.size())
        # import pdb; pdb.set_trace()

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec_vc, context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        scores_vc = self.attn_net_vc(encoder_seq_proj_vc, attn_hidden, t)
        scores_vc_global = self.attn_net_vc_global(encoder_seq_proj_vc, attn_hidden, t)
        scores_vc_lst = [scores_vc, scores_vc_global]

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq if attn_ref is None else attn_ref @ encoder_seq
        context_vec_vc = scores_vc @ encoder_seq_vc if attn_ref_vc is None else attn_ref_vc @ encoder_seq_vc
        context_vec_vc_global = scores_vc_global @ encoder_seq_vc if attn_ref_vc_global is None else attn_ref_vc_global @ encoder_seq_vc

        context_vec = context_vec.squeeze(1)
        context_vec_vc = context_vec_vc.squeeze(1)
        context_vec_vc_global = context_vec_vc_global.squeeze(1)

        context_vec_vc = torch.cat([context_vec_vc, context_vec_vc_global], dim=1)
        context_vec_vc = self.rnn_input_multihead(context_vec_vc)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec_vc, context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc


class Tacotron_pass2_attn(Tacotron_pass2_delib_shareEnc):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True, share_encoder=True, nb_heads_vc=2):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode, 
                 encoder_reduction_factor, encoder_reduction_factor_s, pass2_input, init_model=False, share_encoder=share_encoder)

        # overwrite
        self.nb_heads_vc = nb_heads_vc
        self.decoder = Decoder_pass2_attn(n_mels, decoder_dims, lstm_dims, nb_heads_vc)

        if init_model:
            self.init_model()
            self.num_params()


    def forward(self, x, m, m_p1, s_p1=None, e_p1=None, e_p_p1=None, generate_gta=False, attn_ref=None):
        """
        input
        x [B, Tin]: input text
        m [B, D, Tout]: ref output
        m_p1 [B, D, Tout']: output from the 1st pass
        attn_ref [B, Tout / r, Tin]: reference attention

        output
        m_p2 [B, D, Tout']: output from the 2nd pass
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s))
        m_p1_ds = torch.cat(tmp, dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc_lst = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=attn_ref)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc_lst = [torch.cat(attn_scores_vc, 1) for attn_scores_vc in attn_scores_vc_lst]
        # attn_scores = attn_scores.cpu().data.numpy()

        return (mel_outputs, linear, attn_scores, *attn_scores_vc_lst)


    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, 
        encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_scores_vc_lst = [], [], []
        for i in range(self.nb_heads_vc): attn_scores_vc_lst.append([])

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)

                # print([x.size() for x in scores_vc_lst])
                # print([len(x) for x in attn_scores_vc_lst])
                # print([x is None for x in attn_scores_vc_lst])
                # print([x.append(9) for x in attn_scores_vc_lst])
                # import pdb; pdb.set_trace()
                
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
        elif self.mode=='free_running':
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                             hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break

        assert self.nb_heads_vc==len(scores_vc_lst), f'self.nb_heads_vc {self.nb_heads_vc} != len(scores_vc_lst) {len(scores_vc_lst)}'

        return mel_outputs, attn_scores, attn_scores_vc_lst


    def generate(self, x, m_p1, s_p1=None, e_p1=None, e_p_p1=None, steps=2000):
        self.mode = 'free_running'
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s))
        m_p1_ds = torch.cat(tmp, dim=-1)

        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc_lst = self.decoder_loop(steps, m_p1, go_frame, encoder_seq, encoder_seq_proj, 
            encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc_lst = [torch.cat(attn_scores_vc, 1) for attn_scores_vc in attn_scores_vc_lst]

        # mv to cpu
        linear = linear[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = attn_scores.cpu().data.numpy()[0]
        attn_scores_vc_lst = [attn_scores_vc.cpu().data.numpy()[0] for attn_scores_vc in attn_scores_vc_lst]
        # attn_scores = attn_scores.cpu().data.numpy()

        return (mel_outputs, linear, attn_scores, *attn_scores_vc_lst)


# --------------------------------

class Encoder_vc_taco2(Encoder_vc_taco2_espnet):
    """docstring for Encoder_vc_taco2"""
    def forward(self, xs, ilens=None):
        if ilens is None:
            B, T, D = xs.size()
            device = next(self.parameters()).device
            ilens = torch.ones(B, device=device) * T
        # print(ilens, xs.size())
        # import pdb; pdb.set_trace()
        xs, hlens = super().forward(xs, ilens)
        return xs


class Decoder_pass2_attnAdv(Decoder_pass2_attn):
    def forward(self, encoder_seq, encoder_seq_proj, encoder_seq_vc_lst, encoder_seq_proj_vc_lst, prenet_in, 
        hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=None, attn_ref_vc=None, attn_ref_vc_global=None):
        encoder_seq_vc, encoder_seq_vc_global = encoder_seq_vc_lst
        encoder_seq_proj_vc, encoder_seq_proj_vc_global = encoder_seq_proj_vc_lst

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)
        # print(prenet_out.size(), context_vec.size(), context_vec_vc.size())
        # import pdb; pdb.set_trace()

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec_vc, context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        scores_vc = self.attn_net_vc(encoder_seq_proj_vc, attn_hidden, t)
        scores_vc_global = self.attn_net_vc_global(encoder_seq_proj_vc_global, attn_hidden, t)
        scores_vc_lst = [scores_vc, scores_vc_global]

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq if attn_ref is None else attn_ref @ encoder_seq
        context_vec_vc = scores_vc @ encoder_seq_vc if attn_ref_vc is None else attn_ref_vc @ encoder_seq_vc
        context_vec_vc_global = scores_vc_global @ encoder_seq_vc_global if attn_ref_vc_global is None else attn_ref_vc_global @ encoder_seq_vc_global

        context_vec = context_vec.squeeze(1)
        context_vec_vc = context_vec_vc.squeeze(1)
        context_vec_vc_global = context_vec_vc_global.squeeze(1)

        context_vec_vc = torch.cat([context_vec_vc, context_vec_vc_global], dim=1)
        context_vec_vc = self.rnn_input_multihead(context_vec_vc)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec_vc, context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc


class Tacotron_pass2_attnAdv(Tacotron_pass2_delib_shareEnc):
    """
    separate encoders for each head of attn_vc
    better encoders, designed for speech
    better dim for decoder
    """
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True, share_encoder=True, nb_heads_vc=2, encoder_vc_type='Taco2'):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode, 
                 encoder_reduction_factor[0], encoder_reduction_factor_s[0], pass2_input, init_model=False, share_encoder=share_encoder)

        # overwrite
        self.nb_heads_vc = nb_heads_vc
        self.decoder = Decoder_pass2_attnAdv(n_mels, decoder_dims, lstm_dims, nb_heads_vc)

        # overwrite one encoder, add another
        assert 'y1' in pass2_input or 's1' in pass2_input, 'y1 & s1 are both absent from pass2_input'
        tmp_y1, tmp_s1 = int('y1' in pass2_input), int('s1' in pass2_input)

        tmp = tmp_y1 * n_mels * encoder_reduction_factor[0] + tmp_s1 * decoder_dims * encoder_reduction_factor_s[0]
        if encoder_vc_type=='Taco2':
            self.encoder_vc = Encoder_vc_taco2(tmp, input_layer="linear", embed_dim=embed_dims, elayers=2, eunits=decoder_dims, econv_layers=3, econv_chans=512, econv_filts=5, 
                                                use_batch_norm=True, use_residual=False, dropout_rate=0.5, padding_idx=0)
        elif encoder_vc_type=='Taco1': 
            self.encoder_vc = Encoder_vc(embed_dims, tmp, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj_vc = nn.Linear(decoder_dims, decoder_dims, bias=False)

        tmp = tmp_y1 * n_mels * encoder_reduction_factor[1] + tmp_s1 * decoder_dims * encoder_reduction_factor_s[1]
        if encoder_vc_type=='Taco2':
            self.encoder_vc_global = Encoder_vc_taco2(tmp, input_layer="linear", embed_dim=embed_dims, elayers=2, eunits=decoder_dims, econv_layers=3, econv_chans=512, econv_filts=5, 
                                                use_batch_norm=True, use_residual=False, dropout_rate=0.5, padding_idx=0)
        elif encoder_vc_type=='Taco1': 
            self.encoder_vc_global = Encoder_vc(embed_dims, tmp, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj_vc_global = nn.Linear(decoder_dims, decoder_dims, bias=False)

        self.encoder_reduction_factor = encoder_reduction_factor
        self.encoder_reduction_factor_s = encoder_reduction_factor_s

        if init_model:
            self.init_model()
            self.num_params()


    def forward(self, x, m, m_p1, s_p1=None, e_p1=None, e_p_p1=None, generate_gta=False, attn_ref=None):
        """
        input
        x [B, Tin]: input text
        m [B, D, Tout]: ref output
        m_p1 [B, D, Tout']: output from the 1st pass
        attn_ref [B, Tout / r, Tin]: reference attention

        output
        m_p2 [B, D, Tout']: output from the 2nd pass
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[0]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[0]))
        m_p1_ds = torch.cat(tmp, dim=-1)
        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[1]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[1]))
        m_p1_ds_global = torch.cat(tmp, dim=-1)
        encoder_seq_vc_global = self.encoder_vc_global(m_p1_ds_global)
        encoder_seq_proj_vc_global = self.encoder_proj_vc_global(encoder_seq_vc_global)

        # print(m_p1_ds.size())
        # print(encoder_seq_vc.size(), encoder_seq_proj_vc.size())
        # print(m_p1_ds_global.size())
        # print(encoder_seq_vc_global.size(), encoder_seq_proj_vc_global.size())
        # import pdb; pdb.set_trace()

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc_lst = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, 
            [encoder_seq_vc, encoder_seq_vc_global], [encoder_seq_proj_vc, encoder_seq_proj_vc_global], hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=attn_ref)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc_lst = [torch.cat(attn_scores_vc, 1) for attn_scores_vc in attn_scores_vc_lst]
        # attn_scores = attn_scores.cpu().data.numpy()

        return (mel_outputs, linear, attn_scores, *attn_scores_vc_lst)


    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, 
        encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_scores_vc_lst = [], [], []
        for i in range(self.nb_heads_vc): attn_scores_vc_lst.append([])

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)

                # print([x.size() for x in scores_vc_lst])
                # print([len(x) for x in attn_scores_vc_lst])
                # print([x is None for x in attn_scores_vc_lst])
                # print([x.append(9) for x in attn_scores_vc_lst])
                # import pdb; pdb.set_trace()
                
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
        elif self.mode=='free_running':
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                             hidden_states, cell_states, context_vec, context_vec_vc, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break

        assert self.nb_heads_vc==len(scores_vc_lst), f'self.nb_heads_vc {self.nb_heads_vc} != len(scores_vc_lst) {len(scores_vc_lst)}'

        return mel_outputs, attn_scores, attn_scores_vc_lst


    def generate(self, x, m_p1, s_p1=None, e_p1=None, e_p_p1=None, steps=2000):
        self.mode = 'free_running'
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[0]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[0]))
        m_p1_ds = torch.cat(tmp, dim=-1)
        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[1]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[1]))
        m_p1_ds_global = torch.cat(tmp, dim=-1)
        encoder_seq_vc_global = self.encoder_vc_global(m_p1_ds_global)
        encoder_seq_proj_vc_global = self.encoder_proj_vc_global(encoder_seq_vc_global)

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc_lst = self.decoder_loop(steps, m_p1, go_frame, encoder_seq, encoder_seq_proj, 
            (encoder_seq_vc, encoder_seq_vc_global), (encoder_seq_proj_vc, encoder_seq_proj_vc_global), hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc_lst = [torch.cat(attn_scores_vc, 1) for attn_scores_vc in attn_scores_vc_lst]

        # mv to cpu
        linear = linear[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = attn_scores.cpu().data.numpy()[0]
        attn_scores_vc_lst = [attn_scores_vc.cpu().data.numpy()[0] for attn_scores_vc in attn_scores_vc_lst]
        # attn_scores = attn_scores.cpu().data.numpy()

        return (mel_outputs, linear, attn_scores, *attn_scores_vc_lst)


# --------------------------------

class Tacotron_pass1_smartKV(Tacotron):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 fr_length_ratio=1, share_encoder=True, output_context=True):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode=mode)
        self.fr_length_ratio = fr_length_ratio
        self.share_encoder = share_encoder
        self.output_context = output_context


    def forward(self, x, m, generate_gta=False, generate_fr=False, attn_ref=None):
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta or generate_fr:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores, attn_hiddens, context_vecs = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = m[:, :, t - 1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #         self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                      hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        # qd212, new for Taco_pass1, Concat the mel outputs into sequence
        attn_hiddens = torch.cat(attn_hiddens, dim=2)
        # import pdb; pdb.set_trace()
        # print(attn_hiddens.size())
        # print(len(context_vecs), context_vecs[0].size())
        context_vecs = torch.cat(context_vecs, dim=2)

        # print(len(mel_outputs), mel_outputs[0].size())
        # print(len(attn_hiddens), attn_hiddens[0].size())
        # import pdb; pdb.set_trace()

        ret = (mel_outputs, linear, attn_scores, attn_hiddens)
        if self.share_encoder: ret += (encoder_seq, encoder_seq_proj)
        if self.output_context: ret += (context_vecs,)
        return ret

    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_hiddens, context_vecs = [], [], [], []

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_hiddens.append(hidden_states[0].unsqueeze(-1))
                context_vecs.append(context_vec.unsqueeze(-1))
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                    self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                                 hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1))
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_hiddens.append(hidden_states[0].unsqueeze(-1))
                context_vecs.append(context_vec.unsqueeze(-1))
        elif self.mode=='free_running':
            device = next(self.parameters()).device
            # mask = torch.ones([m.size(0)], device=device)
            # mask = torch.ones([encoder_seq.size(0)], device=device)

            for t in range(0, int(steps*self.fr_length_ratio), self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)

                # mask = mask * (mel_frames >= self.stop_threshold).int().float().prod(-1).prod(-1)
                # tmp = mel_frames * mask.unsqueeze(-1).unsqueeze(-1) + (-4) * (1-mask).unsqueeze(-1).unsqueeze(-1)
                # mel_outputs.append(tmp)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_hiddens.append(hidden_states[0].unsqueeze(-1))
                context_vecs.append(context_vec.unsqueeze(-1))
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break
                # import pdb; pdb.set_trace()

        return mel_outputs, attn_scores, attn_hiddens, context_vecs

    def generate(self, x, steps=2000, m=None, attn_ref=None):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        if self.mode!='free_running':
            import warnings
            msg = f'the mode is not free_running but {self.mode}'
            print(msg)
            warnings.warn(msg)
        mel_outputs, attn_scores, attn_hiddens, context_vecs = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, hidden_states, cell_states, context_vec, attn_ref=attn_ref)

        # # Need a couple of lists for outputs
        # mel_outputs, attn_scores = [], []

        # # Run the decoder loop
        # for t in range(0, steps, self.r):
        #     prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
        #     mel_frames, scores, hidden_states, cell_states, context_vec = \
        #     self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
        #                  hidden_states, cell_states, context_vec, t)
        #     mel_outputs.append(mel_frames)
        #     attn_scores.append(scores)
        #     # Stop the loop if silent frames present
        #     if (mel_frames < self.stop_threshold).all() and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)

        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]

        # qd212, new for Taco_pass1, Concat the mel outputs into sequence
        attn_hiddens = torch.cat(attn_hiddens, dim=2)
        context_vecs = torch.cat(context_vecs, dim=2)
        # no need to put to cpu, as this will be used by Taco_pass2, unless the code runs on cpu

        self.train()

        ret = (mel_outputs, linear, attn_scores, attn_hiddens)
        if self.share_encoder: ret += (encoder_seq, encoder_seq_proj)
        if self.output_context: ret += (context_vecs,)
        return ret


# --------------------------------
class Attention_smartKV(nn.Module):
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):

        # print(encoder_seq_proj.shape)
        # Transform the query vector
        query_proj = self.W(query).unsqueeze(1)

        # Compute the scores
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)

        return scores.transpose(1, 2)

class DotProductAttention(nn.Module):
    
    def __init__(self, attn_dims):
        super().__init__()
        
        self.scale = 1.0/np.sqrt(attn_dims)
        self.softmax = nn.Softmax(dim=2)
       
    def forward(self, keys, query, t, mask=None):
        # query: [B,Q] (hidden state, decoder output, etc.)
        # keys: [B,T,K] (encoder outputs)
        # values: [T,B,V] (encoder outputs)
        # assume Q == K
        
        # compute energy
        query = query.unsqueeze(1) # [B,Q] -> [B,1,Q]
        keys = keys.permute(0,2,1) # [B,T,K] -> [B,K,T]
        energy = torch.bmm(query, keys) # [B,1,Q]*[B,K,T] = [B,1,T]
        energy = self.softmax(energy.mul_(self.scale))
        
        # # apply mask, renormalize
        # energy = energy*mask
        # energy.div(energy.sum(2, keepdim=True))

        # # weight values
        # values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
        # combo = torch.bmm(energy, values).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]

        return energy


class Decoder_pass2_attnAdv_smartKV(Decoder_pass2_attnAdv):
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims, nb_heads_vc=2):
        super().__init__(n_mels, decoder_dims, lstm_dims, nb_heads_vc)
        self.attn_net_vc_global = DotProductAttention(decoder_dims)


    def forward(self, encoder_seq, encoder_seq_proj, encoder_seq_vc_lst, encoder_seq_proj_vc_lst, prenet_in, 
        hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=None, attn_ref_vc=None, attn_ref_vc_global=None, 
        ctx_p1_seq_proj_lst=None, input_mask_x=None, input_mask_y=None):
        encoder_seq_vc, encoder_seq_vc_global = encoder_seq_vc_lst
        encoder_seq_proj_vc, encoder_seq_proj_vc_global = encoder_seq_proj_vc_lst

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)
        # print(prenet_out.size(), context_vec.size(), context_vec_vc.size())
        # import pdb; pdb.set_trace()

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec_vc, context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)
        # Dot product to create the context vector
        context_vec = scores @ encoder_seq if attn_ref is None else attn_ref @ encoder_seq
        context_vec = context_vec.squeeze(1)

        # import pdb; pdb.set_trace()
        # print(encoder_seq_proj.size(), attn_hidden.size(), scores.size())
        # print(ctx_p1_seq_proj_lst[0].size(), context_vec.size())
        # print(ctx_p1_seq_proj_lst[1].size(), context_vec.size())

        scores_vc = self.attn_net_vc(ctx_p1_seq_proj_lst[0], context_vec, t)
        scores_vc_global = self.attn_net_vc_global(ctx_p1_seq_proj_lst[1], context_vec, t) # keys, query
        scores_vc_lst = [scores_vc, scores_vc_global]

        context_vec_vc = scores_vc @ encoder_seq_vc if attn_ref_vc is None else attn_ref_vc @ encoder_seq_vc
        context_vec_vc_global = scores_vc_global @ encoder_seq_vc_global if attn_ref_vc_global is None else attn_ref_vc_global @ encoder_seq_vc_global

        context_vec_vc = context_vec_vc.squeeze(1)
        context_vec_vc_global = context_vec_vc_global.squeeze(1)

        context_vec_vc = torch.cat([context_vec_vc, context_vec_vc_global], dim=1)
        context_vec_vc = self.rnn_input_multihead(context_vec_vc)

        if input_mask_x is not None: context_vec = context_vec * input_mask_x
        if input_mask_y is not None: context_vec_vc = context_vec_vc * input_mask_y

        import pdb; pdb.set_trace()
        print(context_vec.size(), context_vec_vc.size())
        print(context_vec[0,0], context_vec_vc[0,0])

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec_vc, context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc


class Tacotron_pass2_attnAdv_smartKV(Tacotron_pass2_attnAdv):
    """
    separate encoders for each head of attn_vc
    better encoders, designed for speech
    better dim for decoder
    """
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode='teacher_forcing', 
                 encoder_reduction_factor=2, encoder_reduction_factor_s=1, pass2_input='xNy1', init_model=True, share_encoder=True, nb_heads_vc=2, encoder_vc_type='Taco2'):
        super().__init__(embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, mode, 
                 encoder_reduction_factor, encoder_reduction_factor_s, pass2_input, 
                 init_model=False, share_encoder=share_encoder, nb_heads_vc=nb_heads_vc, encoder_vc_type=encoder_vc_type)

        # overwrite
        self.decoder = Decoder_pass2_attnAdv_smartKV(n_mels, decoder_dims, lstm_dims, nb_heads_vc)

        if init_model:
            self.init_model()
            self.num_params()


    def forward(self, x, m, m_p1, s_p1=None, e_p1=None, e_p_p1=None, generate_gta=False, attn_ref=None, c_p1=None, input_mask_x=None, input_mask_y=None):
        """
        input
        x [B, Tin]: input text
        m [B, D, Tout]: ref output
        m_p1 [B, D, Tout']: output from the 1st pass
        attn_ref [B, Tout / r, Tin]: reference attention

        output
        m_p2 [B, D, Tout']: output from the 2nd pass
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        if generate_gta:
            self.eval()
        else:
            self.train()

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[0]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[0]))
        m_p1_ds = torch.cat(tmp, dim=-1)
        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[1]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[1]))
        m_p1_ds_global = torch.cat(tmp, dim=-1)
        encoder_seq_vc_global = self.encoder_vc_global(m_p1_ds_global)
        encoder_seq_proj_vc_global = self.encoder_proj_vc_global(encoder_seq_vc_global)

        ctx_p1_seq_proj_lst = [get_reduced_input_avg(c_p1, self.encoder_reduction_factor_s[i]) for i in range(self.nb_heads_vc)]

        # print(c_p1.size())
        # print(ctx_p1_seq_proj_lst[0].size(), ctx_p1_seq_proj_lst[1].size())
        # print(c_p1[0, 0, :32])
        # print(ctx_p1_seq_proj_lst[0][0,:8,0])
        # print(ctx_p1_seq_proj_lst[1][0,:4,0])
        # import pdb; pdb.set_trace()

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc_lst = self.decoder_loop(steps, m, go_frame, encoder_seq, encoder_seq_proj, 
            [encoder_seq_vc, encoder_seq_vc_global], [encoder_seq_proj_vc, encoder_seq_proj_vc_global], hidden_states, cell_states, context_vec, context_vec_vc, 
            attn_ref=attn_ref, ctx_p1_seq_proj_lst=ctx_p1_seq_proj_lst, input_mask_x=input_mask_x, input_mask_y=input_mask_y)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc_lst = [torch.cat(attn_scores_vc, 1) for attn_scores_vc in attn_scores_vc_lst]
        # attn_scores = attn_scores.cpu().data.numpy()

        return (mel_outputs, linear, attn_scores, *attn_scores_vc_lst)


    def decoder_loop(self, steps, m, go_frame, encoder_seq, encoder_seq_proj, 
        encoder_seq_vc, encoder_seq_proj_vc, hidden_states, cell_states, context_vec, context_vec_vc, attn_ref=None, ctx_p1_seq_proj_lst=None,
        input_mask_x=None, input_mask_y=None):
        # Need a couple of lists for outputs
        mel_outputs, attn_scores, attn_scores_vc_lst = [], [], []
        for i in range(self.nb_heads_vc): attn_scores_vc_lst.append([])

        # Run the decoder loop
        if self.mode=='teacher_forcing':
            for t in range(0, steps, self.r):
                prenet_in = m[:, :, t - 1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, ctx_p1_seq_proj_lst=ctx_p1_seq_proj_lst, 
                                 input_mask_x=input_mask_x, input_mask_y=input_mask_y)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)

                # print([x.size() for x in scores_vc_lst])
                # print([len(x) for x in attn_scores_vc_lst])
                # print([x is None for x in attn_scores_vc_lst])
                # print([x.append(9) for x in attn_scores_vc_lst])
                # import pdb; pdb.set_trace()
                
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
        elif self.mode in ['attention_forcing_online', 'attention_forcing_offline']:
            assert attn_ref is not None, 'in attention_forcing mode, but attn_ref is None'
            for t in range(0, steps, self.r):
                # print(m.size(), attn_ref.size())
                # print(t, steps, self.r)
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                    self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                                 hidden_states, cell_states, context_vec, context_vec_vc, t, attn_ref=attn_ref[:, t//self.r,:].unsqueeze(1), ctx_p1_seq_proj_lst=ctx_p1_seq_proj_lst, 
                                 input_mask_x=input_mask_x, input_mask_y=input_mask_y)
                    # self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                    #              hidden_states, cell_states, context_vec, t, attn_ref=attn_ref[:, t//2,:].unsqueeze(1))
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
        elif self.mode=='free_running':
            for t in range(0, steps, self.r):
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
                mel_frames, scores, scores_vc_lst, hidden_states, cell_states, context_vec, context_vec_vc = \
                self.decoder(encoder_seq, encoder_seq_proj, encoder_seq_vc, encoder_seq_proj_vc, prenet_in,
                             hidden_states, cell_states, context_vec, context_vec_vc, t, ctx_p1_seq_proj_lst=ctx_p1_seq_proj_lst,
                             input_mask_x=input_mask_x, input_mask_y=input_mask_y)
                mel_outputs.append(mel_frames)
                attn_scores.append(scores)
                attn_scores_vc_lst = [attn_scores_vc_lst[i] + [scores_vc_lst[i]] for i in range(self.nb_heads_vc)]
                # Stop the loop if silent frames present
                if (mel_frames < self.stop_threshold).all() and t > 10: break

        assert self.nb_heads_vc==len(scores_vc_lst), f'self.nb_heads_vc {self.nb_heads_vc} != len(scores_vc_lst) {len(scores_vc_lst)}'

        return mel_outputs, attn_scores, attn_scores_vc_lst


    def generate(self, x, m_p1, s_p1=None, e_p1=None, e_p_p1=None, steps=2000, c_p1=None):
        self.mode = 'free_running'
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        # attn_hidden_vc = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        # hidden_states = (attn_hidden, attn_hidden_vc, rnn1_hidden, rnn2_hidden)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)
        context_vec_vc = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        if self.share_encoder:
            encoder_seq = e_p1
            encoder_seq_proj = e_p_p1
        else:
            encoder_seq = self.encoder(x)
            encoder_seq_proj = self.encoder_proj(encoder_seq)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[0]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[0]))
        m_p1_ds = torch.cat(tmp, dim=-1)
        encoder_seq_vc = self.encoder_vc(m_p1_ds)
        encoder_seq_proj_vc = self.encoder_proj_vc(encoder_seq_vc)

        tmp = []
        if 'y1' in self.pass2_input: tmp.append(self.get_reduced_input(m_p1, self.encoder_reduction_factor[1]))
        if 's1' in self.pass2_input: tmp.append(self.get_reduced_input(s_p1, self.encoder_reduction_factor_s[1]))
        m_p1_ds_global = torch.cat(tmp, dim=-1)
        encoder_seq_vc_global = self.encoder_vc_global(m_p1_ds_global)
        encoder_seq_proj_vc_global = self.encoder_proj_vc_global(encoder_seq_vc_global)

        ctx_p1_seq_proj_lst = [get_reduced_input_avg(c_p1, self.encoder_reduction_factor_s[i]) for i in range(self.nb_heads_vc)]

        # Run the decoder loop
        mel_outputs, attn_scores, attn_scores_vc_lst = self.decoder_loop(steps, m_p1, go_frame, encoder_seq, encoder_seq_proj, 
            [encoder_seq_vc, encoder_seq_vc_global], [encoder_seq_proj_vc, encoder_seq_proj_vc_global], hidden_states, cell_states, context_vec, context_vec_vc, 
            attn_ref=None, ctx_p1_seq_proj_lst=ctx_p1_seq_proj_lst)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores_vc_lst = [torch.cat(attn_scores_vc, 1) for attn_scores_vc in attn_scores_vc_lst]

        # mv to cpu
        linear = linear[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()
        attn_scores = attn_scores.cpu().data.numpy()[0]
        attn_scores_vc_lst = [attn_scores_vc.cpu().data.numpy()[0] for attn_scores_vc in attn_scores_vc_lst]
        # attn_scores = attn_scores.cpu().data.numpy()

        return (mel_outputs, linear, attn_scores, *attn_scores_vc_lst)


# --------------------------------

def get_reduced_input_avg(m_p1, encoder_reduction_factor):
    if encoder_reduction_factor > 1:
        B, idim, Lmax = m_p1.shape
        if Lmax % encoder_reduction_factor != 0:
            m_p1 = m_p1[:, :, : -(Lmax % encoder_reduction_factor)]
        m_p1_ds = F.interpolate(m_p1, size=int(Lmax / encoder_reduction_factor), mode='linear', align_corners=True).contiguous()
    else:
        m_p1_ds = m_p1
    return m_p1_ds.transpose(1,2)
#coding=utf8
from turtle import position
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn import Transformer

nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        # embed_size = 768
        # hidden_size = 512
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.tag_embed = nn.Embedding(config.num_tags, config.embed_size)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.transformer = Transformer(d_model=config.embed_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=config.dropout,
                                       batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def positional_encoding(self, X, num_features, dropout_p=0.1, max_len=512):
        r'''
            给输入加入位置编码
        参数：
            - num_features: 输入进来的维度
            - dropout_p: dropout的概率，当其为非零时执行dropout
            - max_len: 句子的最大长度，默认512
        形状：
            - 输入： [batch_size, seq_length, num_features]
            - 输出： [batch_size, seq_length, num_features]

        例子：
            >>> X = torch.randn((2,4,10))
            >>> X = positional_encoding(X, 10)
            >>> print(X.shape)
            >>> torch.Size([2, 4, 10])
        '''

        dropout = nn.Dropout(dropout_p)
        P = torch.zeros((1,max_len,num_features))
        X_ = torch.arange(max_len,dtype=torch.float32).reshape(-1,1) / torch.pow(
            10000,
            torch.arange(0,num_features,2,dtype=torch.float32) /num_features)
        P[:,:,0::2] = torch.sin(X_)
        P[:,:,1::2] = torch.cos(X_)
        X = X + P[:,:X.shape[1],:].to(X.device)
        return dropout(X)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = (1 - batch.tag_mask).bool()
        input_ids = batch.input_ids
        lengths = batch.lengths

        src_embed = self.word_embed(input_ids)
        src_embed = self.positional_encoding(X=src_embed, num_features=self.config.embed_size)
        tgt_embed = self.tag_embed(tag_ids)
        tgt_embed = self.positional_encoding(X=tgt_embed, num_features=self.config.embed_size)

        mask = self.transformer.generate_square_subsequent_mask(max(lengths))
        # packed_inputs = rnn_utils.pack_padded_sequence(src_embed, lengths, batch_first=True)
        # packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        # rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        outs = self.transformer(src=src_embed, tgt=tgt_embed, tgt_mask=mask, src_key_padding_mask=tag_mask, tgt_key_padding_mask=tag_mask)
        # print(outs.shape)
        hiddens = self.dropout_layer(outs)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag != 'O'):
                    print("1111111111")
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += mask.float().unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

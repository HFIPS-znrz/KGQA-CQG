""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, Cross_Transformer, encoder, encoder1, decoder, decoder_1, decoder_2):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder1 = encoder1
        self.decoder1 = decoder_1
        self.decoder2 = decoder_2

        self.contextual_transformer = Cross_Transformer

    def forward(self, src, src1, tgt, tgt1, tgt2, tgt1_index, tgt2_index, lengths, lengths1):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt1 = tgt1[:-1]  # exclude last target from inputs
        tgt2 = tgt2[:-1]  # exclude last target from inputs

        enc_state_1, memory_bank_1, memory_lengths_1 = self.encoder(src, lengths)  # 问题编码
        # enc_state_2, memory_bank_2_new, memory_lengths_2 = self.encoder1(src1, lengths1)  #  @@@@@@@@

        # memory_bank_1, memory_bank_2 = self.contextual_transformer(src, src1)  # 问题  实体

        _ , memory_bank_2_new = self.contextual_transformer(src, src1)  # 问题  实体  !!!!!!!!!

        '''解码关键词'''
        # init_state_for_decoder_1 = [dec_out[i] for i in tgt1_index]
        # init_state_for_decoder_2 = [dec_out[i] for i in tgt2_index]
        #print(inin_state_for_decoder_1)
        self.decoder1.init_state(src1, memory_bank_2_new, None)
        self.decoder2.init_state(src1, memory_bank_2_new, None)

        dec_out_1, attns_1 = self.decoder1(tgt1, memory_bank_2_new, memory_lengths=lengths1)

        dec_out_2, attns_2 = self.decoder2(tgt2, memory_bank_2_new, memory_lengths=lengths1)
        
        '''解码模板'''
        # init_state_for_decoder = torch.tensor([(dec_out_1[i]+dec_out_2[i]).tolist() for i in [0,1]]).to('cuda')
        # init_state_for_decoder = init_state_for_decoder.permute(1,0,2)
        # init_state_for_decoder = [dec_out_1[i]+dec_out_2[i] for i in [0,1]]
        # init_state_for_decoder = torch.cat((dec_out_1, dec_out_2), dim=0).tolist()
        # init_state_for_decoder = torch.tensor([(dec_out_1[i]+dec_out_2[i]).tolist() for i in [0,1]]).to('cuda')
        

        '''问题的编码拼接实体解码器的结果'''
        # t_dec_out_1 = torch.tensor()
        # t_dec_out_2 = []
        # if dec_out_1.size()[0] < dec_out_2.size()[0]:
        #     t_dec_out_1 = torch.zeros(dec_out_2.size()[0], dec_out_2.size()[1], dec_out_2.size()[2]).to('cuda')
        #     t_dec_out_1[:dec_out_1.size()[0], :, :] = dec_out_1
        # else:
        #     t_dec_out_2 = torch.zeros(dec_out_1.size()[0], dec_out_1.size()[1], dec_out_1.size()[2]).to('cuda')
        #     t_dec_out_2[:dec_out_2.size()[0], :, :] = dec_out_2

        '''1'''
        # temp = torch.add(dec_out_1[:2], dec_out_2[:2])
        # memory_bank_1, _ = self.contextual_transformer(memory_bank_1, temp)  # 问题  实体

        '''1.2'''
        # temp = torch.add(dec_out_1[:2], dec_out_2[:2])
        # memory_bank_1, _ = self.contextual_transformer(src, temp)  # 问题  实体

        '''2'''
        # temp = torch.cat((dec_out_1, dec_out_2), dim=0)
        # memory_bank_1_new = torch.cat((temp, memory_bank_1), dim=0)
        # linear = nn.Linear(memory_bank_1_new.size()[0], memory_bank_1.size()[0]).to('cuda')
        # memory_bank_1_new = memory_bank_1_new.permute(1,2,0)
        # memory_bank_1_new = linear(memory_bank_1_new)
        # memory_bank_1 = memory_bank_1_new.permute(2,0,1)

        '''2.2'''
        temp = torch.cat((dec_out_1, dec_out_2), dim=0)
        linear = nn.Linear(temp.size()[0], memory_bank_1.size()[0]).to('cuda')
        temp = temp.permute(1,2,0)
        temp = linear(temp)
        temp = temp.permute(2,0,1)
        memory_bank_1, _ = self.contextual_transformer(memory_bank_1, temp)  # 问题  实体

        '''3'''
        # temp = torch.add(dec_out_1[:2], dec_out_2[:2])
        # expanded_tensor = torch.zeros(memory_bank_1.size()[0], memory_bank_1.size()[1], memory_bank_1.size()[2]).to('cuda')
        # expanded_tensor[:temp.size()[0], :, :] = temp
        # memory_bank_1 = torch.add(expanded_tensor, memory_bank_1)
        
        

        self.decoder.init_state(src, memory_bank_1, None)

        dec_out, attns = self.decoder(tgt, memory_bank_1, memory_lengths=lengths)

        return dec_out, attns, dec_out_1, attns_1, dec_out_2, attns_2

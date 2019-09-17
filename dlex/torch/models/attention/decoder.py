from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlex.utils.logging import logger
from dlex.torch.utils.ops_utils import maybe_cuda, LongTensor


class DecodingStates(dict):
    encoder_outputs: torch.Tensor
    encoder_output_lens: torch.Tensor
    encoder_states: torch.Tensor
    decoder_inputs: torch.Tensor = None
    decoder_outputs: torch.Tensor = None
    attentions: List[torch.Tensor] = None
    decoded_sequences: torch.Tensor = None
    decoded_lengths: List[int] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@dataclass
class BeamSearchConfigs:
    beam_size: int = 4
    penalty: float = 0.
    max_len_ratio: int = 0.5
    min_len_ratio: int = 0
    n_best: int = 1
    

def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection

    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['y_seq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1
    return count == M


def mask_by_length(xs, length, fill=0):
    assert xs.size(0) == len(length)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(length):
        ret[i, :l] = xs[i, :l]
    return ret


def index_select_lm_state(rnnlm_state, dim, vidx):
    if isinstance(rnnlm_state, dict):
        new_state = {}
        for k, v in rnnlm_state.items():
            new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
    elif isinstance(rnnlm_state, list):
        new_state = []
        for i in vidx:
            new_state.append(rnnlm_state[int(i)][:])
    return new_state


class DecoderRNN(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            rnn_type: str, 
            num_layers: int, 
            hidden_size: int, 
            output_size: int, 
            vocab_size: int, 
            attention, 
            sos_idx: int, 
            eos_idx: int, 
            max_length: int, 
            beam_search_configs: BeamSearchConfigs, 
            dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, input_size)
        self.dropout_emb = nn.Dropout(p=dropout)

        self._decoder = nn.ModuleList()
        self._dropout_decoder = nn.ModuleList()
        self._decoder += [
            torch.nn.LSTMCell(hidden_size + input_size, hidden_size) if rnn_type == "lstm"
            else torch.nn.GRUCell(hidden_size + input_size, hidden_size)]
        self._dropout_decoder += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, num_layers):
            self._decoder += [
                torch.nn.LSTMCell(hidden_size, hidden_size) if rnn_type == "lstm"
                else torch.nn.GRUCell(hidden_size + input_size, hidden_size)]
            self._dropout_decoder += [torch.nn.Dropout(p=dropout)]
        self._ignore_id = -1
        self._output = torch.nn.Linear(hidden_size, vocab_size)

        self._attention = attention
        self._sos_idx = sos_idx
        self._eos_idx = eos_idx
        # self.char_list = char_list
        # for label smoothing
        # self.labeldist = labeldist
        # self.vlabeldist = None
        # self.lsm_weight = lsm_weight
        self._input_size = input_size
        self._max_length = max_length
        self._dropout = dropout
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._rnn_type = rnn_type
        self._output_size = vocab_size
        self._logzero = -10000000000.0
        self._beam_search_configs = beam_search_configs

    def zero_state(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        return encoder_outputs.new_zeros(encoder_outputs.size(0), self._hidden_size)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self._rnn_type == "lstm":
            z_list[0], c_list[0] = self._decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in range(1, self._num_layers):
                z_list[l], c_list[l] = self._decoder[l](
                    self._dropout_decoder[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]))
        else:
            z_list[0] = self._decoder[0](ey, z_prev[0])
            for l in range(1, self._num_layers):
                z_list[l] = self._decoder[l](self._dropout_decoder[l - 1](z_list[l - 1]), z_prev[l])
        return z_list, c_list

    @property
    def attention(self):
        return self._attention

    def forward(self, states: DecodingStates, strm_idx=0, use_teacher_forcing=True) -> DecodingStates:
        batch_size = states.encoder_outputs.size(0)
        # ys = [y[y != self.ignore_id] for y in decoder_inputs]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.attention) - 1)

        # initialization
        c_list = [self.zero_state(states.encoder_outputs)]
        z_list = [self.zero_state(states.encoder_outputs)]
        for _ in range(1, self._num_layers):
            c_list.append(self.zero_state(states.encoder_outputs))
            z_list.append(self.zero_state(states.encoder_outputs))
        att_w = None
        z_all = []
        self.attention[att_idx].reset()  # reset pre-computation of h

        if use_teacher_forcing:
            if self._input_size != 0:
                decoder_embedded_inputs = self.dropout_emb(self.embed(states.decoder_inputs[:, :-1]))  # utt x olen x zdim
            else:
                decoder_embedded_inputs = None
            # loop for an output sequence
            for i in range(states.decoder_inputs.size(1) - 1):
                att_c, att_w = self.attention[att_idx](
                    states.encoder_outputs,
                    states.encoder_output_lens,
                    self._dropout_decoder[0](z_list[0]),
                    att_w)
                if decoder_embedded_inputs is not None:
                    ey = torch.cat((
                        decoder_embedded_inputs[:, i, :],
                        att_c), dim=1)  # utt x (input_size + hidden_size)
                else:
                    ey = att_c
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
                z_all.append(self._dropout_decoder[-1](z_list[-1]))
        else:
            max_length = states.decoder_inputs.size(1) if states.decoder_inputs is not None else self._max_length
            for step in range(max_length):
                att_c, att_w = self.attention[att_idx](
                    states.encoder_outputs,
                    states.encoder_output_lens,
                    self._dropout_decoder[0](z_list[0]),
                    att_w)
                if step == 0:
                    embedded_sos = self.dropout_emb(self.embed(
                        maybe_cuda(torch.full((batch_size, ), self._sos_idx, dtype=torch.int64))))
                    decoder_rnn_inputs = torch.cat((embedded_sos, att_c), dim=1)
                else:
                    z_out = self._output(z_all[-1])
                    _, z_out = torch.max(z_out.detach(), dim=1)
                    z_out = self.dropout_emb(self.embed(maybe_cuda(z_out)))
                    decoder_rnn_inputs = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)

                z_list, c_list = self.rnn_forward(decoder_rnn_inputs, z_list, c_list, z_list, c_list)
                z_all.append(self._dropout_decoder[-1](z_list[-1]))

        z_all = torch.stack(z_all, dim=1).view(batch_size, -1, self._hidden_size)
        decoder_outputs = self._output(z_all)
        states.decoder_outputs = decoder_outputs
        states.attentions = None
        return states

    def greedy_search(self, states: DecodingStates):
        states = self.forward(states, use_teacher_forcing=False)
        batch_size = states.encoder_outputs.size(0)
        max_length = states.decoder_inputs.size(1) if states.decoder_inputs is not None else self._max_length
        _, sequences = states.decoder_outputs.max(-1)
        lengths = np.array([max_length] * batch_size)
        for step in range(max_length - 1):
            eos_batches = sequences[:, step].eq(self._eos_idx)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = step

        # acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        # logger.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))
        # attentions = []
        # att_ws = self.calculate_all_attentions(encoder_outputs, encoder_output_lens, decoder_inputs, strm_idx)
        # print(att_ws[0].shape)
        # attentions.append(att_c)
        # print(len(attentions))

        states.decoded_sequences = [seq[:length].cpu().detach().numpy()
                                    for seq, length in zip(sequences, lengths)]
        states.decoded_lengths = lengths
        return states

    def beam_search(self, states: DecodingStates):
        n_best_hypotheses = self.recognize_beam_batch(
            states,
            configs=self._beam_search_configs,
            language_model=None)
        states.decoded_sequences = [n_best[0]['y_seq'][1:-1] for n_best in n_best_hypotheses]
        return states

    def recognize_beam_batch(
            self,
            states: DecodingStates,
            configs: BeamSearchConfigs,
            language_model=None,
            normalize_score=True, strm_idx=0):
        att_idx = min(strm_idx, len(self.attention) - 1)
        encoder_outputs = mask_by_length(states.encoder_outputs, states.encoder_output_lens, 0.0)
        # search params
        batch_size = len(states.encoder_output_lens)

        n_bb = batch_size * configs.beam_size
        n_bo = configs.beam_size * self._output_size
        n_bbo = n_bb * self._output_size
        pad_b = maybe_cuda(torch.LongTensor([i * configs.beam_size for i in range(batch_size)]).view(-1, 1))
        pad_o = maybe_cuda(torch.LongTensor([i * self._output_size for i in range(n_bb)]).view(-1, 1))

        max_encoder_output_len = int(max(states.encoder_output_lens))
        max_len = max(1, int(configs.max_len_ratio * max_encoder_output_len))
        min_len = int(configs.min_len_ratio * max_encoder_output_len)

        # initialization
        c_prev = [maybe_cuda(torch.zeros(n_bb, self._hidden_size)) for _ in range(self._num_layers)]
        z_prev = [maybe_cuda(torch.zeros(n_bb, self._hidden_size)) for _ in range(self._num_layers)]
        c_list = [maybe_cuda(torch.zeros(n_bb, self._hidden_size)) for _ in range(self._num_layers)]
        z_list = [maybe_cuda(torch.zeros(n_bb, self._hidden_size)) for _ in range(self._num_layers)]
        vscores = maybe_cuda(torch.zeros(batch_size, configs.beam_size))

        a_prev = None
        rnn_language_model_prev = None

        self.attention[att_idx].reset()  # reset pre-computation of h

        y_seq = [[self._sos_idx] for _ in range(n_bb)]
        stop_search = [False for _ in range(batch_size)]
        ended_hypotheses = [[] for _ in range(batch_size)]

        exp_encoder_output_lens = maybe_cuda(states.encoder_output_lens).repeat(configs.beam_size).view(-1)
        exp_h = encoder_outputs.unsqueeze(1).repeat(1, configs.beam_size, 1, 1).contiguous()
        exp_h = exp_h.view(-1, exp_h.shape[2], exp_h.shape[3])

        for i in range(max_len):
            last_predicted_labels = LongTensor([y[-1] for y in y_seq])
            last_predicted_label_embeddings = self.dropout_emb(self.embed(last_predicted_labels))
            att_c, att_w = self.attention[att_idx](exp_h, exp_encoder_output_lens, self._dropout_decoder[0](z_prev[0]), a_prev)
            decoder_rnn_inputs = torch.cat((last_predicted_label_embeddings, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(decoder_rnn_inputs, z_list, c_list, z_prev, c_prev)
            local_scores = F.log_softmax(self._output(self._dropout_decoder[-1](z_list[-1])), dim=1)

            # rnn_language_model
            if language_model:
                rnn_language_model_state, local_lm_scores = rnnlm.buff_predict(rnn_language_model_prev, last_predicted_labels, n_bb)
                local_scores = local_scores + self._language_model_weight * local_lm_scores
            local_scores = local_scores.view(batch_size, configs.beam_size, self._output_size)

            if i == 0:
                local_scores[:, 1:, :] = self._logzero
            local_best_scores, local_best_odims = torch.topk(
                local_scores.view(batch_size, configs.beam_size, self._output_size),
                configs.beam_size, 2)

            # local pruning (via xp)
            local_scores = np.full((n_bbo,), self._logzero)
            _best_odims = local_best_odims.view(n_bb, configs.beam_size) + pad_o
            _best_odims = _best_odims.view(-1).cpu().numpy()
            _best_score = local_best_scores.view(-1).cpu().detach().numpy()
            local_scores[_best_odims] = _best_score
            local_scores = maybe_cuda(torch.from_numpy(local_scores).float()).view(batch_size, configs.beam_size, self._output_size)

            # (or indexing)
            # local_scores = to_cuda(self, torch.full((batch, beam, self._output_size), self._logzero))
            # _best_odims = local_best_odims
            # _best_score = local_best_scores
            # for si in range(batch):
            # for bj in range(beam):
            # for bk in range(beam):
            # local_scores[si, bj, _best_odims[si, bj, bk]] = _best_score[si, bj, bk]

            eos_vscores = local_scores[:, :, self._eos_idx] + vscores
            vscores = vscores.view(batch_size, configs.beam_size, 1).repeat(1, 1, self._output_size)
            vscores[:, :, self._eos_idx] = self._logzero
            vscores = (vscores + local_scores).view(batch_size, n_bo)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, configs.beam_size, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self._output_size).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self._output_size) + pad_b).view(-1).data.cpu().tolist()

            y_prev = y_seq[:][:]
            y_seq = [y_seq[l][:] for l in accum_padded_beam_ids]
            for i, token_idx in enumerate(accum_odim_ids):
                y_seq[i].append(token_idx)
            vscores = accum_best_scores
            vidx = LongTensor(accum_padded_beam_ids)

            if isinstance(att_w, torch.Tensor):
                a_prev = torch.index_select(att_w.view(n_bb, *att_w.shape[1:]), 0, vidx)
            elif isinstance(att_w, list):  # multi-head attention
                a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w]
            else:
                # handle the case of location_recurrent when return is a tuple
                a_prev_ = torch.index_select(att_w[0].view(n_bb, -1), 0, vidx)
                h_prev_ = torch.index_select(att_w[1][0].view(n_bb, -1), 0, vidx)
                c_prev_ = torch.index_select(att_w[1][1].view(n_bb, -1), 0, vidx)
                a_prev = (a_prev_, (h_prev_, c_prev_))
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self._num_layers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self._num_layers)]

            if language_model:
                rnn_language_model_prev = index_select_lm_state(rnn_language_model_state, 0, vidx)

            # pick ended hypotheses
            if i > min_len:
                k = 0
                penalty_i = (i + 1) * configs.penalty
                thr = accum_best_scores[:, -1]
                for samp_i in range(batch_size):
                    if stop_search[samp_i]:
                        k = k + configs.beam_size
                        continue
                    for beam_j in range(configs.beam_size):
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            yk.append(self._eos_idx)
                            if len(yk) < 1000: # encoder_output_lens[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                                if normalize_score:
                                    _vscore /= len(yk)
                                _score = _vscore.data.cpu().numpy()
                                ended_hypotheses[samp_i].append({'y_seq': yk, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hypotheses[samp_i], i)
                           for samp_i in range(batch_size)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            # torch.cuda.empty_cache()

        dummy_hypotheses = [{'y_seq': [self._sos_idx, self._eos_idx], 'score': np.array([-float('inf')])}]
        ended_hypotheses = [
            ended_hypotheses[samp_i] if len(ended_hypotheses[samp_i]) != 0
            else dummy_hypotheses for samp_i in range(batch_size)]

        n_best_hypotheses = list(map(
            lambda hypotheses: sorted(hypotheses, key=lambda h: h['score'], reverse=True)[:min(len(hypotheses), configs.n_best)],
            ended_hypotheses))

        return n_best_hypotheses

    def recognize_beam(self, encoder_outputs, configs, rnn_language_model=None, strm_idx=0):
        """
        beam search implementation
        :type configs: BeamSearchConfigs
        :param torch.Tensor h: encoder hidden state (T, eprojs)
        :param Namespace recog_args: argument Namespace containing options
        :param char_list: list of character strings
        :param torch.nn.Module rnn_language_model: language module
        :param int strm_idx: stream index for speaker parallel attention in multi-speaker case
        :return: N-best decoding results
        :rtype: list of dicts
        """
        att_idx = min(strm_idx, len(self.attention) - 1)
        # initialization
        c_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        z_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        for _ in range(1, self._num_layers):
            c_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
            z_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
        self.attention[att_idx].reset()  # reset pre-computation of h
        a = None

        # search parms
        # beam = recog_args.beam_size
        # penalty = recog_args.penalty

        # preprate sos
        y = self._sos_idx
        vy = encoder_outputs.new_zeros(1).long()

        if self._max_len_ratio == 0:
            max_len = encoder_outputs.shape[0]
        else:
            # maxlen >= 1
            max_len = max(1, int(self._max_len_ratio * encoder_outputs.size(0)))
        min_len = int(configs.min_len_ratio * encoder_outputs.size(0))
        logger.info('max output length: ' + str(max_len))
        logger.info('min output length: ' + str(min_len))

        # initialize hypothesis
        if rnn_language_model:
            hyp = {'score': 0.0, 'y_seq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnn_language_model_prev': None}
        else:
            hyp = {'score': 0.0, 'y_seq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        hypotheses = [hyp]
        ended_hypotheses = []

        for i in range(max_len):
            logger.debug('position ' + str(i))

            hypotheses_best_kept = []
            for hyp in hypotheses:
                vy.unsqueeze(1)
                vy[0] = hyp['y_seq'][i]
                ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.attention[att_idx](encoder_outputs.unsqueeze(0), [encoder_outputs.size(0)],
                                                        self._dropout_decoder[0](hyp['z_prev'][0]), hyp['a_prev'])
                ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, hyp['z_prev'], hyp['c_prev'])

                # get n-best local scores and their ids
                local_att_scores = F.log_softmax(self.output(self._dropout_decoder[-1](z_list[-1])), dim=1)
                if rnn_language_model:
                    rnn_language_model_state, local_lm_scores = rnn_language_model.predict(hyp['rnn_language_model_prev'], vy)
                    local_scores = local_att_scores + self._language_model_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(local_scores, self._beam_size, dim=1)

                for j in range(self._beam_size):
                    new_hyp = {'z_prev': z_list[:], 'c_prev': c_list[:], 'a_prev': att_w[:],
                               'score': hyp['score'] + local_best_scores[0, j], 'y_seq': [0] * (1 + len(hyp['y_seq']))}
                    # [:] is needed!
                    new_hyp['y_seq'][:len(hyp['y_seq'])] = hyp['y_seq']
                    new_hyp['y_seq'][len(hyp['y_seq'])] = int(local_best_ids[0, j])
                    if rnn_language_model:
                        new_hyp['rnn_language_model_prev'] = rnn_language_model_state
                    # will be (2 x beam) hypotheses at most
                    hypotheses_best_kept.append(new_hyp)

                hypotheses_best_kept = sorted(
                    hypotheses_best_kept, key=lambda x: x['score'], reverse=True)[:self._beam_size]

            # sort and get nbest
            hypotheses = hypotheses_best_kept

            # add eos in the final loop to avoid that there are no ended hypotheses
            if i == max_len - 1:
                logger.info('adding <eos> in the last position in the loop')
                for hyp in hypotheses:
                    hyp['y_seq'].append(self._eos_idx)

            # add ended hypotheses to a final list, and removed them from current hypotheses
            # (this will be a problem, number of hypotheses < beam)
            remained_hypotheses = []
            for hyp in hypotheses:
                if hyp['y_seq'][-1] == self._eos_idx:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['y_seq']) > min_len:
                        hyp['score'] += (i + 1) * self._penalty
                        if rnn_language_model:  # Word LM needs to add final <eos> score
                            hyp['score'] += self._language_model_weight * rnn_language_model.final(
                                hyp['rnn_language_model_prev'])
                        ended_hypotheses.append(hyp)
                else:
                    remained_hypotheses.append(hyp)

            # end detection
            if end_detect(ended_hypotheses, i) and self._max_len_ratio == 0.0:
                logger.info('end detected at %d', i)
                break

            hypotheses = remained_hypotheses
            if len(hypotheses) > 0:
                logger.debug('remaining hypotheses: ' + str(len(hypotheses)))
            else:
                logger.info('no hypothesis. Finish decoding.')
                break

            for hyp in hypotheses:
                logger.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['y_seq'][1:]]))

            logger.debug('number of ended hypotheses: ' + str(len(ended_hypotheses)))

        n_best_hypotheses = sorted(
            ended_hypotheses, key=lambda x: x['score'], reverse=True)[:min(len(ended_hypotheses), configs.n_best)]

        # check number of hypotheses
        if len(n_best_hypotheses) == 0:
            logger.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(encoder_outputs, recog_args, char_list, rnn_language_model)

        logger.info('total log probability: ' + str(n_best_hypotheses[0]['score']))
        logger.info('normalized log probability: ' + str(n_best_hypotheses[0]['score'] / len(n_best_hypotheses[0]['y_seq'])))

        # remove sos
        return n_best_hypotheses

    def calculate_all_attentions(self, encoder_outputs, encoder_output_lens, decoder_inputs, strm_idx=0):
        """Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index for parallel speaker attention in multi-speaker case
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in decoder_inputs]  # parse padded ys
        att_idx = min(strm_idx, len(self.attention) - 1)

        encoder_output_lens = list(map(int, encoder_output_lens))

        ys_in_pad = decoder_inputs[:, :-1]
        ys_out_pad = decoder_inputs[:, 1:]

        # get length info
        max_length = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(encoder_outputs)]
        z_list = [self.zero_state(encoder_outputs)]
        for _ in range(1, self._num_layers):
            c_list.append(self.zero_state(encoder_outputs))
            z_list.append(self.zero_state(encoder_outputs))
        att_w = None
        att_ws = []
        self.attention[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in range(max_length):
            att_c, att_w = self.attention[att_idx](encoder_outputs, encoder_output_lens, self._dropout_decoder[0](z_list[0]), att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        # att_ws = att_to_numpy(att_ws, self._attention[att_idx])
        return att_ws
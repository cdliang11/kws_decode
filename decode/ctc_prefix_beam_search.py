# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from collections import defaultdict

import torch

from decode.utils import log_add


def top_k_by_value(input_dict, k):
    top_k_items = heapq.nlargest(k, input_dict.items(), key=lambda item : item[1])
    # 提取键列表
    keys = [item[0] for item in top_k_items]
    # 提取值列表
    values = [item[1] for item in top_k_items]
    return keys, values


class DecodeResult:
    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None,
                 sequence_confidence: List[List[float]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times
        self.sequence_confidence = sequence_confidence


class PrefixScore:
    """ For CTC prefix beam search """
    def __init__(self,
                 s: float = float('-inf'),
                 ns: float = float('-inf'),
                 v_s: float = float('-inf'),
                 v_ns: float = float('-inf')):
        self.s = s  # blank_ending_score
        self.ns = ns  # none_blank_ending_score
        self.v_s = v_s  # viterbi blank ending score
        self.v_ns = v_ns  # viterbi none blank ending score
        self.cur_token_prob = float('-inf')  # prob of current token
        self.times_s = []  # times of viterbi blank path
        self.times_ns = []  # times of viterbi none blank path

    def score(self):
        return log_add([self.s, self.ns])

    def viterbi_score(self):
        return self.v_s if self.v_s > self.v_ns else self.v_ns

    def times(self):
        return self.times_s if self.v_s > self.v_ns else self.times_ns

def ctc_prefix_beam_search(ctc_probs: torch.Tensor, ctc_lens: torch.Tensor,
                           beam_size: int, keywords_idx: List[int] = None) -> List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    batch_size = ctc_probs.shape[0]
    results = []
    # CTC prefix beam search can not be paralleled, so search one by one
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        cur_hyps = [(tuple(), PrefixScore(0.0, -float('inf'), 0.0, 0.0))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())

            # filter keyword
            if keywords_idx is not None and len(keywords_idx) > 0:
                filter_logp = {}
                for idx in keywords_idx:
                    filter_logp[idx] = logp[idx]
                top_k_index, top_k_logp = top_k_by_value(filter_logp, beam_size)
            else:
                # 2.1 First beam prune: select topk best
                top_k_logp, top_k_index_tmp = logp.topk(beam_size)  # (beam_size,)
                top_k_index = [id.item() for id in top_k_index_tmp]

            for u in top_k_index:
                # u = u.item()
                prob = logp[u].item()
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == 0:  # blank
                        next_score = next_hyps[prefix]
                        next_score.s = log_add([next_score.s,
                                               prefix_score.score() + prob])
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                    elif u == last:
                        #  Update *uu -> *u;
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add([next_score1.ns,
                                                 prefix_score.ns + prob])
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.vs_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy(
                                )
                                next_score1.times_ns[-1] = t

                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u, )
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add([next_score2.ns,
                                                 prefix_score.s + prob])
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(t)
                    else:
                        n_prefix = prefix + (u, )
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add([next_score.ns,
                                                prefix_score.score() + prob])
                        if next_score.v_ns < prefix_score.viterbi_score(
                        ) + prob:
                            next_score.v_ns = prefix_score.viterbi_score(
                            ) + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(t)
            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: x[1].score(),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]

        nbest = [y[0] for y in cur_hyps]
        nbest_scores = [y[1].score() for y in cur_hyps]
        nbest_times = [y[1].times() for y in cur_hyps]
        best = nbest[0]
        best_score = nbest_scores[0]
        best_time = nbest_times[0]

        results.append(
            DecodeResult(tokens=best,
                         score=best_score,
                         times=best_time,
                         nbest=nbest,
                         nbest_scores=nbest_scores,
                         nbest_times=nbest_times,
                         sequence_confidence=None))
    return results

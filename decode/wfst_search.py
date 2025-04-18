# Copyright (c) 2025 Chengdong Liang (liangchengdongd@qq.com)
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

import numpy as np
import copy
from pynini import Fst
import math

char_map = {2: 0, 1: 1, 3: 1112, 4: 1593, 5: 5239, 6: 5968}


class LatticArc:

    def __init__(self, ilabel, olabel, weight, nextstate):
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight
        self.nextstate = nextstate


class Token:
    """Token used in token passing decode algorithm.
    The token can be linked by another token or None.
    """

    def __init__(self):
        self.active = False
        self.is_filler = True,
        self.score = 0.0  # 存储解码图上路径得分 log域相加
        self.num_keyword_frames = 0  # 统计解码图上路径上的唤醒词帧数
        self.average_keyword_score = 0.0  # 唤醒词平均得分 score / 帧数 （忽略blank）
        self.keyword = 0  # 唤醒词id
        self.num_frames_of_current_state = 0  # 当前状态自循环的帧数
        self.num_keyword_states = 0  # 唤醒词的state数
        self.max_score_of_current_state = 0.0  # 当前state的最大得分
        self.average_max_keyword_score = 0.0  # 平均最大得分
        self.average_max_keyword_score_before = 0.0  # 上一个state的平均最大得分
        self.keyword_score = []

    def reset(self):
        self.active = False
        self.is_filler = True
        self.score = 0.0
        self.num_keyword_frames = 0
        self.average_keyword_score = 0.0
        self.keyword = 0
        self.num_frames_of_current_state = 0
        self.num_keyword_states = 0
        self.max_score_of_current_state = 0.0
        self.average_max_keyword_score = 0.0
        self.average_max_keyword_score_before = 0.0
        self.keyword_score = []

    def update(self, prev, olabel, is_self_arc, is_filler, is_blank, am_score,
               time):
        if not self.active or (self.active
                               and self.score < prev.score + am_score
                               ):  # and self.score < prev.score + am_score):
            # it's a keyword state
            if not is_filler:
                self.score = prev.score + am_score
                t = prev.num_keyword_frames
                self.average_keyword_score = (
                    am_score + prev.average_keyword_score * t) / (t + 1)
                self.num_keyword_frames = t + 1
                self.keyword_score = copy.deepcopy(prev.keyword_score)
                if not is_blank:
                    if is_self_arc:
                        self.num_frames_of_current_state = prev.num_frames_of_current_state + 1  # 统计自旋的次数
                        self.num_keyword_states = prev.num_keyword_states
                        self.max_score_of_current_state = max(
                            prev.max_score_of_current_state, am_score)
                        self.average_max_keyword_score_before = prev.average_max_keyword_score
                        self.keyword_score[
                            -1] = self.max_score_of_current_state

                    else:
                        self.num_frames_of_current_state = 1
                        self.num_keyword_states = prev.num_keyword_states + 1
                        self.max_score_of_current_state = am_score
                        self.average_max_keyword_score_before = prev.average_max_keyword_score
                        self.keyword_score.append(
                            self.max_score_of_current_state)
                    self.average_max_keyword_score = (
                        self.max_score_of_current_state +
                        prev.average_max_keyword_score_before *
                        (prev.num_keyword_states - 1)
                    ) / self.num_keyword_states
                    if olabel != 0:
                        self.keyword = olabel
                else:
                    # it's a blank state
                    self.num_frames_of_current_state = prev.num_frames_of_current_state
                    self.num_keyword_states = prev.num_keyword_states
                    self.max_score_of_current_state = prev.max_score_of_current_state
                    self.average_max_keyword_score_before = prev.average_max_keyword_score
                    self.average_max_keyword_score = prev.average_max_keyword_score
        self.active = True
        self.is_filler = is_filler


class WfstDecoder:

    def __init__(self, fst_file, filler_table=None):
        self.fst = Fst.read(fst_file)
        self.filler_table = filler_table
        self.num_frames = 0
        self.min_keyword_frames = 0
        self.min_frames_for_last_state = 0

        self.max_tokenpassing_frames = 100

        self.prev_tokens = [Token() for _ in range(self.fst.num_states())]
        self.cur_tokens = [Token() for _ in range(self.fst.num_states())]

        self.reset()

    def end_detect(self):
        if self.cur_tokens and self.num_steps_decoded >= self.max_seq_len:
            return True
        else:
            return False

    def reset(self):
        for i, tok in enumerate(self.prev_tokens):
            tok.reset()
        for i, tok in enumerate(self.cur_tokens):
            tok.reset()
        self.prev_tokens[0].active = True
        self.num_frames = 0

    def is_filler(self, token):
        return token == 1

    def is_final(self, state):
        return state == 9

    def spot(self, ctc_probs, time):
        """
        ctc_probs: [vocab_size]
        """
        legal = False
        for i, tok in enumerate(self.prev_tokens):
            if tok.active:
                for arc in self.fst.arcs(i):
                    if arc.ilabel == 0 or arc.ilabel == 2:  # eps and blank
                        # non-emitting arc
                        am_score = 0.0
                        is_filler = self.is_filler(arc.ilabel)
                        is_self_arc = i == arc.nextstate
                        olabel = arc.olabel
                        self.cur_tokens[arc.nextstate].update(
                            tok, olabel, is_self_arc, is_filler, True,
                            am_score, time)
                    else:
                        # emitting arc
                        am_score = ctc_probs[char_map[arc.ilabel]]  # log
                        is_filler = self.is_filler(arc.ilabel)
                        is_self_arc = i == arc.nextstate
                        olabel = arc.olabel
                        self.cur_tokens[arc.nextstate].update(
                            tok, olabel, is_self_arc, is_filler, False,
                            am_score, time)
        # find best score
        best_state, best_final_state = 0, 0
        best_score, best_final_score = self.cur_tokens[0].score, 0.0
        reach_final = False
        for i, tok in enumerate(self.cur_tokens):
            if tok.active and best_score < tok.score:
                best_score = tok.score
                best_state = i
            if tok.active and self.is_final(i):
                if not reach_final:
                    best_final_score = tok.score
                    best_final_state = i
                    reach_final = True
                elif best_final_score < tok.score:
                    best_final_score = tok.score
                    best_final_state = i

        if reach_final:
            confidence = math.exp(
                self.cur_tokens[best_final_state].average_max_keyword_score)
            keyword = self.cur_tokens[best_final_state].keyword
            if self.cur_tokens[
                    best_final_state].num_keyword_frames > self.min_keyword_frames and self.cur_tokens[
                        best_final_state].num_frames_of_current_state > self.min_frames_for_last_state and confidence > 0.5:
                legal = True
            else:
                legal = False

        self.prev_tokens = copy.deepcopy(self.cur_tokens)
        for i, tok in enumerate(self.cur_tokens):
            tok.reset()

        self.num_frames += 1
        if self.num_frames > self.max_tokenpassing_frames and self.prev_tokens[
                best_state].is_filler:
            for i, tok in enumerate(self.prev_tokens):
                tok.reset()

        return legal

    def decode(self, ctc_probs):
        print(ctc_probs.shape)  # [1, 54, 6000]  logsoftmax
        print(self.fst.num_states())
        ctc_probs = ctc_probs[0].cpu().numpy()  # [54, 6000]

        for t in range(ctc_probs.shape[0]):
            legal = self.spot(ctc_probs[t], t)
            if legal:
                print("time: ", t, "wake up")


if __name__ == "__main__":
    fst_file = "data/TL.fst"
    decoder = WfstDecoder(fst_file)
    ctc_probs = np.load("data/nihaodawei.npy")

    results = decoder.decode(ctc_probs)

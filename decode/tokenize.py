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

class tokenize:
    def __init__(self, unit_file):
       self.unit2id = {}
       self.id2unit = {}
       with open(unit_file, 'r') as f:
           for line in f:
               unit, id = line.strip().split()
               self.unit2id[unit] = int(id)
               self.id2unit[int(id)] = unit

    def detokenize(self, ids):
        tokens = []
        for id in ids:
            tokens.append(self.id2unit[id])
        return tokens


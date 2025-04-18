

import torch
import numpy as np

from decode.ctc_prefix_beam_search import ctc_prefix_beam_search
from decode.wfst_search import WfstDecoder
from decode.tokenize import tokenize


if __name__ == "__main__":
    ctc_probs = np.load("data/nihaodawei.npy")
    print(ctc_probs.shape)
    ctc_probs = torch.from_numpy(ctc_probs)  # [batch, len, dim]
    ctc_lens = torch.tensor([ctc_probs.shape[1]] * ctc_probs.shape[0])  # [batch]
    print(ctc_lens)

    tokenizer = tokenize("data/units.txt")

    # results = ctc_prefix_beam_search(ctc_probs, ctc_lens, beam_size=10)
    # # print(results.nbest)
    # for result in results:
    #     for beam in result.nbest:
    #         print(tokenizer.detokenize(beam))

    fst_file = "data/TL.fst"
    decoder = WfstDecoder(fst_file)
    results = decoder.decode(ctc_probs)



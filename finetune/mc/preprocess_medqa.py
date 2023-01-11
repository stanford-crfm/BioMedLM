import os
import json
import random
import shutil
import numpy as np
from tqdm import tqdm


root = "data/mc"
os.system(f"mkdir -p {root}")


def dump_jsonl(data, fpath):
    with open(fpath, "w") as outf:
        for d in data:
            print (json.dumps(d), file=outf)

def process_medqa(fname):
    dname = "medqa_usmle"
    lines = open(f"raw_data/medqa/data_clean/questions/US/4_options/phrases_no_exclude_{fname}.jsonl").readlines()
    outs, lens = [], []
    for i, line in enumerate(tqdm(lines)):
        stmt = json.loads(line)
        sent1 = stmt["question"]
        ends = [stmt["options"][key] for key in "ABCD"]
        outs.append({"id": f"{fname}-{i:05d}",
                      "sent1": sent1,
                      "sent2": "",
                      "ending0": ends[0],
                      "ending1": ends[1],
                      "ending2": ends[2],
                      "ending3": ends[3],
                      "label": ord(stmt["answer_idx"]) - ord("A")
                    })
        lens.append(len(sent1) + max([len(ends[0]),len(ends[1]), len(ends[2]), len(ends[3])]))
    print ("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
    #
    os.system(f'mkdir -p {root}/{dname}_hf')
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")


process_medqa("train")
process_medqa("test")
process_medqa("dev")

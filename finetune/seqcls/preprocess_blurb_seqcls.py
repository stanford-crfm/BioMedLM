import os
import csv
import json
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm


def dump_jsonl(data, fpath):
    with open(fpath, "w") as outf:
        for d in data:
            print (json.dumps(d), file=outf)


######################### BLURB sequence classification #########################
root = "data"
os.system(f"mkdir -p {root}")


def process_pubmedqa(fname):
    dname = "pubmedqa"
    print (dname, fname)
    if fname in ["train", "dev"]:
        data = json.load(open(f"raw_data/blurb/data_generation/data/pubmedqa/pqal_fold0/{fname}_set.json"))
    elif fname == "test":
        data = json.load(open(f"raw_data/blurb/data_generation/data/pubmedqa/{fname}_set.json"))
    else:
        assert False
    outs, lens = [], []
    for id in data:
        obj = data[id]
        context = " ".join([c.strip() for c in obj["CONTEXTS"] if c.strip()])
        question = obj["QUESTION"].strip()
        label = obj["final_decision"].strip()
        assert label in ["yes", "no", "maybe"]
        outs.append({"id": id, "sentence1": question, "sentence2": context, "label": label})
        lens.append(len(question) + len(context))
    print ("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
    #
    os.system(f"mkdir -p {root}/{dname}_hf")
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")

process_pubmedqa("test")
process_pubmedqa("train")
process_pubmedqa("dev")


def process_bioasq(fname):
    dname = "bioasq"
    print (dname, fname)
    df = pd.read_csv(open(f"raw_data/blurb/data_generation/data/BioASQ/{fname}.tsv"), sep="\t", header=None)
    outs, lens = [], []
    for _, row in df.iterrows():
        id       = row[0].strip()
        question = row[1].strip()
        context  = row[2].strip()
        label    = row[3].strip()
        assert label in ["yes", "no"]
        outs.append({"id": id, "sentence1": question, "sentence2": context, "label": label})
        lens.append(len(question) + len(context))
    print ("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
    #
    os.system(f"mkdir -p {root}/{dname}_hf")
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")

process_bioasq("test")
process_bioasq("dev")
process_bioasq("train")

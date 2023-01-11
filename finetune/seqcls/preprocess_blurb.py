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
root = "data/seqcls"
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



def process_blurb_RE(dname, fname):
    print (dname, fname)
    fpath = f"raw_data/blurb/data_generation/data/{dname}/{fname}.tsv"
    os.system(f"mkdir -p {root}/{dname}_hf")
    lens = []
    with open(f"{root}/{dname}_hf/{fname}.json", "w") as outf:
        for i, line in enumerate(open(fpath)):
            if i==0 and dname == "GAD":
                continue
            line = line.strip()
            id, sent, label = line.split("\t")
            if "false" in label:
                label = "0"
            out = {"id": id, "sentence": sent, "label": label}
            print (json.dumps(out), file=outf)
            lens.append(len(sent))
    print ("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))

process_blurb_RE("chemprot", "test")
process_blurb_RE("chemprot", "dev")
process_blurb_RE("chemprot", "train")

process_blurb_RE("DDI", "test")
process_blurb_RE("DDI", "train")
process_blurb_RE("DDI", "dev")

process_blurb_RE("GAD", "test")
process_blurb_RE("GAD", "dev")
process_blurb_RE("GAD", "train")



def process_BIOSSES(fname):
    dname = "BIOSSES"
    print (dname, fname)
    fpath = f"raw_data/blurb/data_generation/data/{dname}/{fname}.tsv"
    os.system(f"mkdir -p {root}/{dname}_hf")
    lens = []
    with open(f"{root}/{dname}_hf/{fname}.json", "w") as outf:
        for i, line in enumerate(open(fpath)):
            line = line.strip()
            if i==0:
                continue
            id, sent1, sent2, label = line.split("\t")
            out = {"sentence1": sent1, "sentence2": sent2, "label": float(label), "id": id}
            print (json.dumps(out), file=outf)
            lens.append(len(sent1) + len(sent2))
    print ("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))

process_BIOSSES("test")
process_BIOSSES("dev")
process_BIOSSES("train")


def process_HOC(fname):
    dname = "HoC"
    print (dname, fname)
    fpath = f"raw_data/blurb/data_generation/data/{dname}/{fname}.tsv"
    os.system(f"mkdir -p {root}/{dname}_hf")
    lens = []
    with open(f"{root}/{dname}_hf/{fname}.json", "w") as outf:
        cur_abs_id = ""
        cur_abs = ""
        cur_label = [0] * 10
        for i, line in enumerate(open(fpath)):
            if i==0:
                continue
            line = line.strip()
            label, sent, id = line.split("\t")
            label = [int(elm.split("_")[1]) for elm in label.split(",")]
            if id.split("_")[0] != cur_abs_id:
                if cur_abs:
                    out = {"sentence": cur_abs, "label": cur_label, "id": cur_abs_id}
                    print (json.dumps(out), file=outf)
                    lens.append(len(cur_abs))
                    cur_abs_id = ""
                    cur_abs = ""
                    cur_label = [0] * 10
            cur_abs_id = id.split("_")[0]
            cur_abs = (cur_abs +" "+ sent).strip()
            cur_label = [int(l1 | l2) for l1, l2 in zip(label, cur_label)]
        if cur_abs:
            out = {"sentence": cur_abs, "label": cur_label, "id": cur_abs_id}
            print (json.dumps(out), file=outf)
            lens.append(len(cur_abs))
    print ("total", len(lens), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))

process_HOC("train")
process_HOC("test")
process_HOC("dev")




######################### BLURB token classification #########################
root = "data/tokcls"
os.system(f"mkdir -p {root}")


def process_blurb_NER(dname, fname):
    print (dname, fname)
    fpath = f"raw_data/blurb/data_generation/data/{dname}/{fname}.tsv"
    outs = []
    lens = []
    total = 0
    if True:
        tokens, labels = [], []
        id = 0
        for i, line in enumerate(open(fpath)):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if tokens:
                    outs.append({"tokens": tokens, "ner_tags": labels, "id": str(id)})
                    lens.append(len(" ".join(tokens)))
                    total += len([l for l in labels if l.startswith("B")])
                    tokens, labels = [], []
                    id += 1
                continue
            token, label = line.split("\t")
            tokens.append(token)
            labels.append(label)
        if tokens:
            outs.append({"tokens": tokens, "ner_tags": labels, "id": str(id)})
            lens.append(len(" ".join(tokens)))
            total += len([l for l in labels if l.startswith("B")])
            id += 1
    print ("total", total, "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
    #
    os.system(f"mkdir -p {root}/{dname}_hf")
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")


process_blurb_NER("BC2GM", "train")
process_blurb_NER("BC2GM", "test")
process_blurb_NER("BC2GM", "dev")

process_blurb_NER("BC5CDR-disease", "train")
process_blurb_NER("BC5CDR-disease", "dev")
process_blurb_NER("BC5CDR-disease", "test")

process_blurb_NER("BC5CDR-chem", "train")
process_blurb_NER("BC5CDR-chem", "dev")
process_blurb_NER("BC5CDR-chem", "test")

process_blurb_NER("NCBI-disease", "train")
process_blurb_NER("NCBI-disease", "dev")
process_blurb_NER("NCBI-disease", "test")

process_blurb_NER("JNLPBA", "train")
process_blurb_NER("JNLPBA", "dev")
process_blurb_NER("JNLPBA", "test")



def process_PICO(dname, fname):
    print (dname, fname)
    fpath = f"raw_data/blurb/data_generation/data/{dname}/{fname}.tsv"
    outs = []
    lens = []
    total = 0
    if True:
        tokens, labels = [], []
        id = 0
        for i, line in enumerate(open(fpath)):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if tokens:
                    outs.append({"tokens": tokens, "ner_tags": labels, "id": str(id)})
                    lens.append(len(" ".join(tokens)))
                    total += len([l for l in labels if l!="O"])
                    tokens, labels = [], []
                    id += 1
                continue
            token, label = line.split("\t")
            if label.startswith("I"):
                label = "B" + label[1:]
            tokens.append(token)
            labels.append(label)
        if tokens:
            outs.append({"tokens": tokens, "ner_tags": labels, "id": str(id)})
            lens.append(len(" ".join(tokens)))
            total += len([l for l in labels if l!="O"])
            id += 1
    print ("total", total, "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
    #
    os.system(f"mkdir -p {root}/{dname}_hf")
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")


process_PICO("ebmnlp", "train")
process_PICO("ebmnlp", "dev")
process_PICO("ebmnlp", "test")

## Setting Up BLURB (PubMedQA and BioASQ)

1.) Download [BioASQ](http://www.bioasq.org/) and [PubMedQA](https://pubmedqa.github.io/) original data. Make sure when downloading and expanding the data that it matches these paths: `raw_data/blurb/data_generation/data/pubmedqa` and `raw_data/blurb/data_generation/data/BioASQ` in this directory. For more details, review the `preprocess_blurb_seqcls.py` script to see the specific paths the preprocessing script expects.

2.) Run the `preprocess_medqa.py` script in this directory to produce the data in the format expected by our fine-tuning code.

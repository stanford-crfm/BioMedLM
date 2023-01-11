## Setting Up MedQA

1.) Download data from [MedQA GitHub](https://github.com/jind11/MedQA) . The GitHub should have a link to a Google Drive. Make sure to download the contents to a directory path matching `raw_data/medqa/data_clean/questions/US/4_options` in this directory. For more details, review the `preprocess_medqa.py` script to see the specific paths the preprocessing script expects.

2.) Run the `preprocess_medqa.py` script in this directory to produce the data in the format expected by our fine-tuning code.

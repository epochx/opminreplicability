Code for the paper "Replication issues in syntax-based aspect extraction for opinion mining"

- Download and install the software needed
    1. download and install Senna, http://ronan.collobert.com/senna/
    2. download and install CoreNLP 3.6, http://stanfordnlp.github.io/CoreNLP/history.html
    3. download and install pyFIM http://www.borgelt.net/pyfim.html
    4. `pip install scipy numpy Levenshtein`
    
- Download data and create environment
    1. `create_data_folder.sh path_where_to_create_data_folder`
    2. modify ./enlp/settings.py accordingly
    
- Pre-process datasets
    1. `python process_corpus.py`

- Run (will take several hours depending on the number of cores available)
    1. `python run.py path_to_store_output_json_files`


Check `process_corpus.py --help` and `run.py --help` for more details on how run them.
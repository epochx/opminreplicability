# Code for the paper "Replication issues in syntax-based aspect extraction for opinion mining"

- Download and install the software needed
    1. Download and install Senna, http://ronan.collobert.com/senna/
    2. Download and install CoreNLP 3.6, http://stanfordnlp.github.io/CoreNLP/history.html
      - Needs Java 8
    3. Download and install pyFIM http://www.borgelt.net/pyfim.html
      - Download `fim.so` and place it in python's `dist-packages` directory
    4. Run `pip install scipy numpy python-Levenshtein`

- Download data and create environment
    1. Make `create_data_folder.sh` executable: `chmod +x create_data_folder.sh`
    2. Run `./create_data_folder.sh path/to/data/folder` (No trailing slash!)
    3. Modify `./enlp/settings.py` accordingly

- Pre-process datasets
    1. `python process_corpus.py`

- Run (will take several hours depending on the number of cores available)
    1. `python run.py path/to/store/output/json/files`


Check `process_corpus.py --help` and `run.py --help` for more details on how run them.

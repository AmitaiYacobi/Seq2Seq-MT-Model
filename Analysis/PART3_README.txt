=====================
Name: Amitai Yacobi
ID: 316418300
=====================

Before you run the script, make sure you have the "seaborn" package installed by the following command:

pip3 install seaborn

==============================
Running the script:
For running the script, open a terminal in this directory and type the following command:

python3 train.py path/to/train.src path/to/train.trg path/to/dev.src path/to/dev.trg path/to/test.src path/to/test.trg

For example - 

python3 train.py ../data/train.src ../data/train.trg ../data/dev.src ../data/dev.trg ../data/test.src ../data/test.trg   

==============================
Visualization:
All of the heatmaps visualizations are located in the directory "plots"

==============================
Weights:
The attention weights for the chosen example are located in the directory "attention_weights"
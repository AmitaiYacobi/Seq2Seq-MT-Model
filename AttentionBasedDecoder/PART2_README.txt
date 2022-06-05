=====================
Name: Amitai Yacobi
ID: 316418300
=====================

Train script:
For running the train script, open a terminal in this directory and type the following command:

python3 train.py path/to/train.src path/to/train.trg path/to/dev.src path/to/dev.trg path/to/test.src path/to/test.trg

For example - 

python3 train.py ../data/train.src ../data/train.trg ../data/dev.src ../data/dev.trg ../data/test.src ../data/test.trg   


======================

Evaluation script:
After you ran the train script, you can run the script by opening a terminal in this directory and typing the following command:

python3 test.py path/to/test.src path/to/test.trg

For example - 

python3 test.py ../data/test.src ../data/test.trg   

======================

If you want to run the evaluation on my best results without training the model from start, type the following command:

python3 test.py path/to/test.src path/to/test.trg -b yes

For example - 

python3 test.py ../data/test.src ../data/test.trg -b yes
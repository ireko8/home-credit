Hello!

Below you can find a outline of how to reproduce my solution for the Home Credit Default Risk competition.
If you run into any trouble with the setup/code or have any questions please contact me at pornoromen0101@gmail.com

#Directory Structure
code_doc
├── README.md
├── requirement.txt
├─run.sh
├─setting
│   └── keras.json
└── src
    ├── config.py
    ├── input
    ├── main.py
    ├── neural_net.py
    ├── preprocess.py
    └── utils.py

#ARCHIVE CONTENTS
code_doc: original kaggle model upload - contains original code
src : code to generate DAE features from ONODERA's LB804 features.
requirement.txt: version of python packages
setting: keras setting dir

#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS (128 GB boot disk)
CPU: Intel(R) Xeon(R) CPU E5462 @ 2.80GHz
GPU: 1 x NVIDIA GTX1080
Mem: 32GB

#SOFTWARE:
Python 3.6.2
CUDA 9.1
cuddn 7.1.1
nvidia drivers v.390.48

#DATA PROCESSING
0. put ONODERA's LB804 features in src/input.
1. run run.sh (e.g. command "bash run.sh")

You can generate *.ftr file in the "src/output/"
Format type is feather.
# EINNs: Epidemiologically-Informed Neural Networks

## Pre-print paper

Implementation of the paper "EINNs: Epidemiologically-Informed Neural Networks."

Authors: Alexander Rodríguez, Jiaming Cui, Bijaya Adhikari, Naren Ramakrishnan, B. Aditya Prakash

Pre-print: [https://arxiv.org/abs/2202.10446](https://arxiv.org/abs/2202.10446)

## Requirements

Use the package manager [conda](https://docs.conda.io/en/latest/) to install required Python dependencies. Note: We used Python 3.7.

```bash
conda env create -f requirements.yml
```

## Training

The following command will train and predict for all regions from epidemic week 202036 to 202109:

```bash
python main.py --region AL --dev cpu --exp 400 --start_ew 202036 --end_ew 202109 --step 2
```

More examples can be found in ```run.sh```.

You can set up your own model hyperparameter values (e.g. learning rate, loss weights) in the file ```./setup/EINN-params.json```.

## Contact:

If you have any questions about the code, please contact Alexander Rodriguez at arodriguezc[at]gatech[dot]edu and/or B. Aditya Prakash badityap[at]cc[dot]gatech[dot]edu 



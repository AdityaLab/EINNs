# [AAAI-23] EINNs: Epidemiologically-Informed Neural Networks

## Publication

Implementation of the paper "EINNs: Epidemiologically-informed Neural Networks" published in AAAI 2023.

Authors: Alexander Rodríguez, Jiaming Cui, Naren Ramakrishnan, Bijaya Adhikari, B. Aditya Prakash

Paper + appendix: [https://arxiv.org/abs/2202.10446](https://arxiv.org/abs/2202.10446)

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

## Cite our work
If you find our work useful, please cite our work:
- Alexander Rodríguez, Jiaming Cui, Naren Ramakrishnan, Bijaya Adhikari, B. Aditya Prakash. 2023. EINNs: Epidemiologically-informed Neural Networks. In Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37.

```
@inproceedings{rodriguez2022einns,
  title={EINNs: Epidemiologically-Informed Neural Networks},
  author={Rodr'\iguez, Alexander and Cui, Jiaming and Ramakrishnan, Naren and Adhikari, Bijaya and Prakash, B. Aditya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  year={2023}
}

# Neural Autoregressive Flows for Markov Boundary Learning (FANS)

This is the implementation for our paper: [Neural Autoregressive Flows for Markov Boundary Learning](https://arxiv.org/pdf/2407.04992), accepted at IEEE ICDM 2025.

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.12-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/pdf/2407.04992"><img src="https://img.shields.io/badge/arXiv-2407.07973-b31b1b.svg" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#files">Structure</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

## Setup 

```bash
conda env create -n fans --file env.yml
conda activate fans
```

## Files

- data/: Create synhetic datasets, Masking mechanisms. 
- data_gen/: Save data here. 
- model/: Source code of models. 
- model_save/: Save model checkpoints here.
- result_save/: Save results here.
- utils/: Code utils.
- metrics.py: Metrics for evaluation.
- run_linear.py: Run linear data.
- run_nonlinear.py: Run nonlinear data.

## Experiments

### Linear data

**Default:** Data sampled from an ER graph with 100 nodes, expected degree 1, 5000 samples, and Gaussian noise.

```
python run_linear.py
```

### Nonlinear data

**Default:** Data are sampled from an ER graph with 30 nodes and expected degree 1, using 1000 samples. The data-generating process is drawn from a Gaussian Process, with additive Gaussian noise.

Train FANS:
```
python run_nonlinear.py --train --mode flow --d 30 --data_seed 42
```

## Citation

```
@inproceedings{nguyen2025neural,
      title={Neural Autoregressive Flows for Markov Boundary Learning}, 
      author={Nguyen, Khoa and Duong, Bao and Huynh, Viet and Nguyen, Thin},
      year={2025},
      booktitle = {IEEE International Conference on Data Mining},
}
```

# MP3

MP3 (Multi-Period Pattern Pre-training) is a plug-and-play pre-training plugin for spatio-temporal graph neural networks (STGNNs) to distinguish temporal mirages caused by incomplete period observation and heterogeneous spatial correlations. It learns multi-period temporal patterns, captures global spatial relations via a memory bank, and models cross-period dependencies through a causality-enhanced Transformer. Evaluated on five real-world datasets across five STGNN baselines, MP3 consistently improves forecasting performance, reducing MAE by 4.7% and RMSE by 5.0% on average.

## Requirements

`python3`, `torch`, `numpy`, `scipy`, `tensorflow`

## Hardware environment

NVIDIA RTX 3090

## Datasets

The California dataset is downloaded from the Caltrans Performance Measurement System ([PeMS](https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=3&submit=Submit)) website, we use Station 5-Minute traffic speed data ranging from 2022-04-01 to 2022-06-31. We provide the processed California [dataset link](https://drive.google.com/file/d/1p75j3JqHMT00DpiBH7x7WAnJeX1njVxr/view?usp=sharing) for public use.

## Usage

### Preprocessing

To preprocess the long historical traffic time series:

```bash
python preprocess/preprocess.py --dataset <DATASET_NAME>
```

Supported datasets: `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`, `CA`.

### Training with MP3 Plugin

All downstream baselines support unified command-line arguments. Use `--dataset` to specify the dataset and `--use_long` to enable the MP3 plugin.

**Example (with plugin):**

```bash
python MSDR/run.py --dataset PEMS04 --use_long
python ST_WA/run.py --dataset PEMS08 --use_long
python GWnet/run.py --dataset CA --use_long
python STGCN/run.py --dataset PEMS03 --use_long
```

**Example (without plugin):**

```bash
python MSDR/run.py --dataset PEMS04
python ST_WA/run.py --dataset PEMS08
python GWnet/run.py --dataset CA
python STGCN/run.py --dataset PEMS03
```

## Acknowledgement

We thank the authors of the following repositories for code reference:

- [BigST](https://github.com/uci-dsp-lab/BigST)
- [TimesNet](https://github.com/thuml/Time-Series-Library)
- [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet)
- [BasicTS](https://github.com/zezhishao/BasicTS)
- [Nodeformer](https://github.com/qitianwu/NodeFormer)
- [Performer](https://github.com/lucidrains/performer-pytorch)

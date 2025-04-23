# F2CGT

`F2CGT` is a fast GNN training system incorporating feature compression. It reduces the size of feature vectors and lets on-GPU cache keep more compressed features so as to substantially reduces the volume of data transferred between the CPU and GPU. It additionally enables caching graph structure in GPU for speeding up sampling, and avoids graph partitioning for eliminating cross-machine communication in distributed training. 

* `Feature compression`: We propose a two-level, hybrid feature compression approach that applies different compression methods to different graph nodes. This differentiated choice strikes a balance between rounding errors, compression ratios, model accuracy loss, and  preprocessing costs. Our theoretical analysis proves that this approach offers convergence and comparable model accuracy as the conventional training without feature compression.
* `On-GPU cache`: We also co-design the on-GPU cache sub-system with compression-enabled training. The new cache sub-system, driven by a cost model, runs new cache policies to carefully choose graph nodes with high access frequencies, and well partitions the spare GPU memory for various types of graph data, for improving cache hit rates.
* `Decompression and Aggregation Fusion`: We also introduce an online feature decompression step, which converts compressed features into their original size, and then passes decompressed data to the aggregation computation of GNN training. The decompressed input features occupy a major portion of the GPU memory space, ranging from hundreds of MBs to several GBs. Based on this observation, we further fuse the decompression operator and the aggregation operator into a single operator. This brings two benefits. First, it improves the computation efficiency by reducing overhead such as kernel launching. Second, it eliminates the need to allocate GPU memory for storing intermediate results after decompression, leaving more GPU memory available for the cache system.

`F2CGT` supports scalar, vector and two-level quantizations.

[Paper in VLDB2024](https://dl.acm.org/doi/10.14778/3681954.3681968)

[Supplementary material](https://github.com/gpzlx1/F2CGT-supplemental)

# Install
We use `conda` to manage our python environment.

* Install mamba

  ```shell
  ## prioritize 'conda-forge' channel
  conda config --add channels conda-forge
  
  ## update existing packages to use 'conda-forge' channel
  conda update -n base --all
  
  ## install 'mamba'
  conda install -n base mamba

  ## init 'mamba'
  mamba shell init --shell bash --root-prefix=~/.local/share/mamba
  
  ## new env
  mamba create -n raft python=3.10

  ## activate
  mamba activate raft
  ```

* Install raft

  ```shell
  # for CUDA 12.4
  mamba install -c rapidsai -c conda-forge -c nvidia libraft-headers==23.10 libraft==23.10 pylibraft==23.10 cuda-version=12.4
  ```

* Install DGL and PyTorch

  ```shell
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
  pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
  pip install pybind11
  ```

* Modify "raft/cluster/detail/kmeans.cuh"
  ```c++
  // disable all logger::get(RAFT_NAME).set_level(params.verbosity); in "raft/cluster/detail/kmeans.cuh"
  // logger::get(RAFT_NAME).set_level(params.verbosity);
  ```
  * Reason: https://github.com/rapidsai/raft/issues/2357

* Install F2CGT

   ```shell
   git clone --recursive git@github.com:gpzlx1/F2CGT.git

   # install third_party
   cd F2CGT/third_party/ShmTensor
   # replace 8.6 with your device sm ability
   bash install.sh

   # install F2CGT
   cd F2CGT
   bash install.sh
   ```

## Usage

1. Prepare datasets:

   See [examples/process_dataset.py](./examples/process_dataset.py). You can also use a custom script to preprocess other datasets.

   The processed dataset format is:

   ```shell
   # demo for products
   python3 examples/process_dataset.py --dataset ogbn-products --root /data --out-dir datasets/

   # it generates the following files in datasets
   .
   ├── csc_indices.pt
   ├── csc_indptr.pt
   ├── features.pt
   ├── hotness.pt
   ├── labels.pt
   ├── meta.pt
   ├── seeds.pt  # used for two-level quantizations
   ├── test_nids.pt
   ├── train_nids.pt
   └── valid_nids.pt
   ```

   Each of the `*.pt` file is a pytorch `tensor`.

2. Compress graph:

   See [examples/process_compress.py](./examples/process_compress.py). Usage:

   ```shell
   # demo for products with two-level quantization
   torchrun --nproc_per_node 2 examples/process_compress.py  --root ./datasets/ --target-compression-ratio 128 --n-clusters 256 --with-seeds --with-feature
   ```

3. Start training:
     ```shell
     # demo for training graph on products with 2 GPUs
     torchrun  --nproc_per_node 2 examples/train_graphsage_nodeclass.py --num-hidden 128 --fan-out 12,12,12 --num-epochs 21 --eval-every 20 --root ./datasets/ --compress-root ./datasets/ --fusion --create-cache
     ```

More scripts about training can be found in `scripts`.

# Known Issues

* We recommend using 8 GPUs for better compression quality. While 1 GPU can complete the compression, you'll need to adjust the block slice method in scripts to maintain the same quality.
* In fact, the current implementation of compression is unreasonable—it should be decoupled from the number of GPUs, but this hasn't been implemented yet and will be addressed later. If this feature is urgently needed, please file an issue.

# Citation
```latex
@article{10.14778/3681954.3681968,
author = {Ma, Yuxin and Gong, Ping and Wu, Tianming and Yi, Jiawei and Yang, Chengru and Li, Cheng and Peng, Qirong and Xie, Guiming and Bao, Yongcheng and Liu, Haifeng and Xu, Yinlong},
title = {Eliminating Data Processing Bottlenecks in GNN Training over Large Graphs via Two-level Feature Compression},
year = {2024},
issue_date = {July 2024},
publisher = {VLDB Endowment},
volume = {17},
number = {11},
issn = {2150-8097},
url = {https://doi.org/10.14778/3681954.3681968},
doi = {10.14778/3681954.3681968},
abstract = {Training GNNs over large graphs faces a severe data processing bottleneck, involving both sampling and feature loading. To tackle this issue, we introduce F2CGT, a fast GNN training system incorporating feature compression. To avoid potential accuracy degradation, we propose a two-level, hybrid feature compression approach that applies different compression methods to various graph nodes. This differentiated choice strikes a balance between rounding errors, compression ratios, model accuracy loss, and preprocessing costs. Our theoretical analysis proves that this approach offers convergence and comparable model accuracy as the conventional training without feature compression. Additionally, we also co-design the on-GPU cache sub-system with compression-enabled training within F2CGT. The new cache sub-system, driven by a cost model, runs new cache policies to carefully choose graph nodes with high access frequencies, and well partitions the spare GPU memory for various types of graph data, for improving cache hit rates. Finally, extensive evaluation of F2CGT on two popular GNN models and four datasets, including three large public datasets, demonstrates that F2CGT achieves a compression ratio of up to 128 and provides GNN training speedups of 1.23-2.56\texttimes{} and 3.58--71.46\texttimes{} for single-machine and distributed training, respectively, with up to 32 GPUs and marginal accuracy loss.},
journal = {Proc. VLDB Endow.},
month = {aug},
pages = {2854–2866},
numpages = {13}
}
```

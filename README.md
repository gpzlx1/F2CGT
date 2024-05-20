# F2CGT

`F2CGT` is a fast GNN training system incorporating feature compression. It reduces the size of feature vectors and lets on-GPU cache keep more compressed features so as to substantially reduces the volume of data transferred between the CPU and GPU. It additionally enables caching graph structure in GPU for speeding up sampling, and avoids graph partitioning for eliminating cross-machine communication in distributed training. 

* `Feature compression`: We propose a two-level, hybrid feature compression approach that applies different compression methods to different graph nodes. This differentiated choice strikes a balance between rounding errors, compression ratios, model accuracy loss, and  preprocessing costs. Our theoretical analysis proves that this approach offers convergence and comparable model accuracy as the conventional training without feature compression.
* `On-GPU cache`: We also co-design the on-GPU cache sub-system with compression-enabled training. The new cache sub-system, driven by a cost model, runs new cache policies to carefully choose graph nodes with high access frequencies, and well partitions the spare GPU memory for various types of graph data, for improving cache hit rates.
* `Decompression and Aggregation Fusion`: We also introduce an online feature decompression step, which converts compressed features into their original size, and then passes decompressed data to the aggregation computation of GNN training. The decompressed input features occupy a major portion of the GPU memory space, ranging from hundreds of MBs to several GBs. Based on this observation, we further fuse the decompression operator and the aggregation operator into a single operator. This brings two benefits. First, it improves the computation efficiency by reducing overhead such as kernel launching. Second, it eliminates the need to allocate GPU memory for storing intermediate results after decompression, leaving more GPU memory available for the cache system.

`F2CGT` supports scalar, vector and two-level quantizations.

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
  
  ## new env
  mamba create -n raft-23.10 python=3.10
  ```

* Install raft

  ```shell
  # for CUDA 11.8
  mamba install -c rapidsai -c conda-forge -c nvidia libraft-headers==23.10 libraft==23.10 pylibraft==23.10 cuda-version=11.8
  ```

* Install DGL and PyTorch

  ```shell
  # python3 3.10.14
  # raft 23.10
  mamba install -c dglteam/label/cu118 dgl==2.0
  mamba install pybind11
  ```

* Install F2CGT

   ```shell
   git clone --recursive git@github.com:gpzlx1/F2CGT.git

   # install third_party
   cd F2CGT/third_party/ShmTensor
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

# F2CGT

`F2CGT` is a fast GNN training system incorporating feature compression. It reduces the size of feature vectors and lets on-GPU cache keep more compressed features so as to substantially reduces the volume of data transferred between the CPU and GPU. It additionally enables caching graph structure in GPU for speeding up sampling, and avoids graph partitioning for eliminating cross-machine communication in distributed training. 

* `Feature compression`: We propose a two-level, hybrid feature compression approach that applies different compression methods to different graph nodes. This differentiated choice strikes a balance between rounding errors, compression ratios, model accuracy loss, and  preprocessing costs. Our theoretical analysis proves that this approach offers convergence and comparable model accuracy as the conventional training without feature compression.
* `On-GPU cache`: We also co-design the on-GPU cache sub-system with compression-enabled training. The new cache sub-system, driven by a cost model, runs new cache policies to carefully choose graph nodes with high access frequencies, and well partitions the spare GPU memory for various types of graph data, for improving cache hit rates.

Extensive evaluation of `F2CGT` on two popular GNN models and four datasets, including three large public datasets, demonstrates that `F2CGT` achieves a compression ratio of up to 64 and provides GNN training speedups of 1.17-1.88x and 3.58-19.59x for single-machine and distributed training, respectively, with up to 32 GPUs and marginal accuracy loss.

# Install
## Software Version
* Ubuntu 20.04
* CUDA v11.8
* PyTorch v2.1.2
* DGL v1.1.3

## Install HADES
We use `pip` to manage our python environment.

1. Install PyTorch and DGL
2. Download `F2CGT` source code
   ```shell
   git clone https://github.com/gpzlx1/F2CGT
   ```
3. Download submodules
   ```shell
   cd F2CGT
   git submodule update --init --recursive
   ```
4. Install `F2CGT`
   ```shell
   bash install.sh
   ```

## Usage

1. Prepare datasets:

   See [example/preprocess_datagen.py](./example/preprocess_datagen.py). You can also use a custom script to preprocess other datasets.

   The processed dataset format is:

   ```
   .
   ├── features.pt
   ├── indices.pt
   ├── indptr.pt
   ├── labels.pt
   ├── metadata.pt
   ├── test_idx.pt
   ├── train_idx.pt
   └── valid_idx.pt
   ```

   Each of the `*.pt` file is a pytorch `tensor`.

2. Compress graph:

   See [example/preprocess_compress.py](./example/preprocess_compress.py). Usage:

   ```shell
   torchrun --nproc_per_node ${#GPUs} example/preprocess_compress.py \ 
     --save-path ${path to save the compressed graph} \
     --root ${path to the graph generated in step 1} \
     --methods sq,vq \
     --configs "[{'target_bits':4},{'width':32,'length':4096}]" \
     --dataset ${dataset name} \
     --num-gpus ${#GPUs} \
     --fan-out 12,12,12 \
     --batch-size 1000 \
     --compress-batch-sizes 1000000,1000000
   ```

   * `methods` is the compression methods for train/valid/test nodes and other nodes, respectively.
   * `configs` is the configures for the compression of train/valid/test nodes and other nodes. Its input format is the same as python's `list` and `dict`.
   For detailed information about compression configure, you can refer to our paper.
   * `fan-out` is the fanouts of your training task. This arg is for presampling.
   * `batch-size` is the batch size of your training task.
   * `compress-batch-sizes` is the number of nodes of feature sample and processing feature batch during compression.
   * `num-gpus` is the number of GPUs participated in the compression process. We will distribute the compression task to multi GPUs to reduce the compresssion latency.

3. Compute the slopes required by on-GPU cache:

   Edit [example/ip_config.txt](example/ip_config.txt), and add the IPs of all the machines that will participate in the training (they can access each other by SSH without a password). For example:

   ```shell
   python3 example/preprocess_compute_slope.py \
     --dataset ${dataset name} \
     --root ${path to the compressed graph generated in step 2} \
     --num-trainers ${#GPUs} \
     --fan-out 12,12,12 \
     --batch-size 1000,1000,1000 \
     --adj-step 0.05 \
     --adj-epochs 10 \
     --feat-step 0.2 \
     --feat-epochs 5
   ```

   * `--fan-out` is the fanouts of your training task.
   * `--batch-size` is the batch size of your training task.
   * `--adj-step` & `--feat-step`: we compute the slope by counting the cache hit times and reduced training latency. To do so, we will launch a few presampling epochs, each of which will cache different ratios of the graph structure (or feature). Cache ratio will improve from 0, `--*--step` indicates the improving step.
   * `--adj-epochs` & `--feat-epochs` indicate the number of computing epochs. For example, if `--feat-step==0.2` and `--feat-epochs==5`, then 5 epochs will be performed and their feature cache ratios are: `0, 0.2, 0.4, 0.6, 0.8`. 

   The output will be:

   ```shell
   ====================================
   Graph adj slope: xxxxxx
   Feature slope: xxxxxx
   Compute slopes time: xxxxxx sec
   ====================================
   ```

   `Graph adj slope` and `Feature slope` represent the average reduced latency of sampling and feature loading, respectively. They should be used as input args in the following training steps.

3. Start training:

   * Single-machine training:

     ```shell
     python3 example/train_graphsage_nodeclassification.py \ 
       --num-trainers ${#GPUs} \
       --num-epochs ${#training epochs} \
       --num-hidden ${model hidden dim} \
       --dropout ${model dropout} \
       --lr ${model learning rate} \
       --root ${path to the compressed graph generated in step 2} \
       --feat-slope ${feature slope} \
       --adj-slope ${graph adj slope} \
       --fan-out ${sampling fanouts} \
       --batch-size ${training batch size}
     ```

   * Distributed training:

     Each machine has a replica of the entire compressed graph.

     ```shell
     torchrun --nproc_per_node ${#GPUs per machine} \
       --master_port 12345 \
       --nnodes ${#machines} \
       --node_rank ${rank of this machine node} \
       --master_addr ${IP address of the master machine node} \
       example/train_dist_graphsage_nodeclassification.py --num-trainers 8 \
        --num-trainers ${#GPUs per machine} \
        --num-epochs ${#training epochs} \
        --num-hidden ${model hidden dim} \
        --dropout ${model dropout} \
        --lr ${model learning rate} \
        --root ${path to the compressed graph generated in step 2} \
        --feat-slope ${feature slope} \
        --adj-slope ${graph adj slope} \
        --fan-out ${sampling fanouts} \
        --batch-size ${training batch size}
     ```

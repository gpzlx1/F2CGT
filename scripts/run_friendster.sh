#!/bin/bash

# friendster-sq32-sage
echo "friendster-sq32-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/friendster/ --target-compression-ratio 32
mkdir ../v2_dataset/friendster/sq32
mv ../v2_dataset/friendster/compression_* ../v2_dataset/friendster/sq32/
# train
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/friendster/ --compress-root ../v2_dataset/friendster/sq32/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

friendster-vq64-sage
echo "friendster-vq64-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/friendster/ --target-compression-ratio 64
mkdir ../v2_dataset/friendster/vq64
mv ../v2_dataset/friendster/compression_* ../v2_dataset/friendster/vq64/
# train
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/friendster/ --compress-root ../v2_dataset/friendster/vq64/ --num-gpus $i --batch-size 1000 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

# friendster-sq8,vq128-sage
echo "friendster-sq8,vq128-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/friendster/ --target-compression-ratio 128 --with-seeds
mkdir ../v2_dataset/friendster/sq8,vq128
mv ../v2_dataset/friendster/compression_* ../v2_dataset/friendster/sq8,vq128/
mv ../v2_dataset/friendster/seeds_compression_data.pt ../v2_dataset/friendster/sq8,vq128/
# train
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/friendster/ --compress-root ../v2_dataset/friendster/sq8,vq128/ --num-gpus $i --batch-size 1000 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model sage --reserved-mem 3
done

# friendster-sq32-gat
echo "friendster-sq32-gat"
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/friendster/ --compress-root ../v2_dataset/friendster/sq32/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --reserved-mem 2.0 --num-heads 8
done

# friendster-vq64-gat
echo "friendster-vq64-gat"
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/friendster/ --compress-root ../v2_dataset/friendster/vq64/ --num-gpus $i --batch-size 1000 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --reserved-mem 2.0 --num-heads 8
done

# friendster-sq8,vq128-gat
echo "friendster-sq8,vq128-gat"
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/friendster/ --compress-root ../v2_dataset/friendster/sq8,vq128/ --num-gpus $i --batch-size 1000 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --reserved-mem 2.0 --num-heads 8
done
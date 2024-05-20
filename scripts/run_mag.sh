#!/bin/bash

# mag240m-sq8,vq128-sage
echo "mag240m-sq8,vq128-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/mag240m/ --target-compression-ratio 128 --with-seeds
mkdir ../v2_dataset/mag240m/sq8,vq128
mv ../v2_dataset/mag240m/compression_* ../v2_dataset/mag240m/sq8,vq128/
mv ../v2_dataset/mag240m/seeds_compression_data.pt ../v2_dataset/mag240m/sq8,vq128/
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/mag240m/ --compress-root ../v2_dataset/mag240m/sq8,vq128/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 5,10,15 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 3.0
done
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/mag240m/ --compress-root ../v2_dataset/mag240m/sq8,vq128/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 5,10,15 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 3.0
done
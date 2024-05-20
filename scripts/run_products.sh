#!/bin/bash

# products-sq32-sage
echo "products-sq32-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/products/ --target-compression-ratio 32
mkdir ../v2_dataset/products/sq32
mv ../v2_dataset/products/compression_* ../v2_dataset/products/sq32/
# train
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq32/ --num-gpus 8 --batch-size 1535 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
for i in 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq32/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

# products-vq64-sage
echo "products-vq64-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/products/ --target-compression-ratio 64
mkdir ../v2_dataset/products/vq64
mv ../v2_dataset/products/compression_* ../v2_dataset/products/vq64/
# train
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/vq64/ --num-gpus 8 --batch-size 1535 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
for i in 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/vq64/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

# products-sq8,vq128-sage
echo "products-sq8,vq128-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/products/ --target-compression-ratio 128 --with-seeds
mkdir ../v2_dataset/products/sq8,vq128
mv ../v2_dataset/products/compression_* ../v2_dataset/products/sq8,vq128/
mv ../v2_dataset/products/seeds_compression_data.pt ../v2_dataset/products/sq8,vq128/
# train
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq8,vq128/ --num-gpus 8 --batch-size 1535 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
for i in 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq8,vq128/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

# products-sq32-gat
echo "products-sq32-gat"
# train
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq32/ --num-gpus 8 --batch-size 1535 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
for i in 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq32/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
done

# products-vq64-gat
echo "products-vq64-gat"
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/vq64/ --num-gpus 8 --batch-size 1535 --eval-every 2 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
for i in 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/vq64/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
done

# products-sq8,vq128-gat
echo "products-sq8,vq128-gat"
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq8,vq128/ --num-gpus 8 --batch-size 1535 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
for i in 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq8,vq128/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
done

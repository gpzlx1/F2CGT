!/bin/bash

# papers-sq32-sage
echo "papers-sq32-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/papers/ --target-compression-ratio 32
mkdir ../v2_dataset/papers/sq32
mv ../v2_dataset/papers/compression_* ../v2_dataset/papers/sq32/
# train
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq32/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

# papers-vq64-sage
echo "papers-vq64-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/papers/ --target-compression-ratio 64
mkdir ../v2_dataset/papers/vq64
mv ../v2_dataset/papers/compression_* ../v2_dataset/papers/vq64/
# train
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/vq64/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 6 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

# papers-sq8,vq128-sage
echo "papers-sq8,vq128-SAGE"
# compress
torchrun --nproc_per_node 8 examples/process_compress.py --n-clusters 256 --root ../v2_dataset/papers/ --target-compression-ratio 64  --with-feature --with-seeds
mkdir ../v2_dataset/papers/sq8,vq64
mv ../v2_dataset/papers/compression_* ../v2_dataset/papers/sq8,vq64/
mv ../v2_dataset/papers/seeds_compression_data.pt ../v2_dataset/papers/sq8,vq64/
torchrun --nproc_per_node 8 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq32/ --num-gpus 8 --batch-size 1536 --eval-every 2 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage --num-heads 8 --reserved-mem 4
# train
for i in 8 4 2 1
do
    torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq8,vq128/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
done

papers-sq32-gat
echo "papers-sq32-gat"
# train
for i in 8 4 2 1
do
   torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq32/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
done

# papers-vq64-sage
echo "papers-vq64-SAGE"
for i in 8 4 2 1
do
   torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/vq64/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.5
done

# papers-sq8,vq128-sage
echo "papers-sq8,vq128-SAGE"
for i in 8 4 2 1
do
   torchrun --nproc_per_node $i examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq8,vq128/ --num-gpus $i --batch-size 1536 --eval-every 999 --fan-out 20,20,20 --num-hidden 128 --num-epochs 4 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.5
done

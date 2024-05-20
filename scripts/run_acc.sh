torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq32/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/vq64/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq8,vq128/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq32/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/vq64/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage --reserved-mem 2.0
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq8,vq128/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model sage --reserved-mem 2.5

torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq32/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/vq64/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/products/ --compress-root ../v2_dataset/products/sq8,vq128/ --num-gpus 2 --batch-size 1536 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq32/ --num-gpus 2 --batch-size 1000 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/vq64/ --num-gpus 2 --batch-size 1000 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 2.0
torchrun --nproc_per_node 2 examples/train_graphsage_nodeclass.py --root ../v2_dataset/papers/ --compress-root ../v2_dataset/papers/sq8,vq128/ --num-gpus 2 --batch-size 1000 --eval-every 10 --fan-out 20,20,20 --num-hidden 128 --num-epochs 100 --breakdown --fusion --create-cache --model gat --num-heads 8 --reserved-mem 3.5
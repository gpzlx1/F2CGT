torchrun --nproc_per_node 8 example/preprocess_compress.py --save-path ~/workspace/compressed_dataset/mag240m/0/ --configs "[{'target_bits':4},{'width':32,'length':4096}]" --methods sq,vq --root ~/workspace/processed_dataset/mag240M/ --dataset mag240M --num-gpus 8 --fan-out 12,12,12 --compress-batch-sizes 500000,50000 | tee ~/workspace/compress_log/2023-12-27-mag240m-0.log
torchrun --nproc_per_node 8 example/preprocess_compress.py --save-path ~/workspace/compressed_dataset/ogbn-products/1/ --configs "[{'target_bits':4},{'width':32,'length':16384}]" --methods sq,vq --root ~/workspace/processed_dataset/ogbn-products/ --dataset ogbn-products --num-gpus 8 --fan-out 12,12,12 --compress-batch-sizes 1000000,1000000 | tee ~/workspace/compress_log/2023-12-27-products-1.log
torchrun --nproc_per_node 8 example/preprocess_compress.py --save-path ~/workspace/compressed_dataset/ogbn-papers100M/1/ --configs "[{'target_bits':4},{'width':32,'length':16384}]" --methods sq,vq --root ~/workspace/processed_dataset/ogbn-papers100M/ --dataset ogbn-papers100M --num-gpus 8 --fan-out 12,12,12 --compress-batch-sizes 1000000,500000 | tee ~/workspace/compress_log/2023-12-27-papers100M-1.log
torchrun --nproc_per_node 8 example/preprocess_compress.py --save-path ~/workspace/compressed_dataset/friendster/1/ --configs "[{'target_bits':4},{'width':32,'length':16384}]" --methods sq,vq --root ~/workspace/processed_dataset/friendster/ --dataset friendster --num-gpus 8 --fan-out 12,12,12 --compress-batch-sizes 1000000,100000 | tee ~/workspace/compress_log/2023-12-27-friendster-1.log

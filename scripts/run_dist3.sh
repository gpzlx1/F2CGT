ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_graphsage_nodeclassification.py --num-trainers 8  --num-epochs 20 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-products/2 --eval-every 21 --feat-slope 0.000611 --adj-slope 0.080265 --fan-out 12,12,12"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_gat_nodeclassification.py --num-trainers 8  --num-epochs 20 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-products/2 --eval-every 21 --feat-slope 0.000611 --adj-slope 0.080265 --fan-out 12,12,12"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_graphsage_nodeclassification.py --num-trainers 8  --num-epochs 7 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_gat_nodeclassification.py --num-trainers 8  --num-epochs 7 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_graphsage_nodeclassification.py --num-trainers 8  --num-epochs 6 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/friendster/2 --dataset friendster --eval-every 21 --feat-slope 0.000622 --adj-slope 0.066833 --fan-out 12,12,12 --reserved-mem 3.0 --batch-size 1024"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_gat_nodeclassification.py --num-trainers 8  --num-epochs 6 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/friendster/2 --dataset friendster --eval-every 21 --feat-slope 0.000622 --adj-slope 0.066833 --fan-out 12,12,12 --reserved-mem 3.5 --batch-size 1024"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_graphsage_nodeclassification.py --num-trainers 8  --num-epochs 7 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/mag240m/2 --dataset mag240M --eval-every 21 --feat-slope 0.000532 --adj-slope 0.025124 --fan-out 10,25 --reserved-mem 2.0"
ssh 172.31.21.164 "source ~/workspace/venv/bin/activate && cd ~/workspace/Project-Q/ && torchrun --nproc_per_node 8 --master_port 12345 --nnodes 4 --node_rank 3 --master_addr 172.31.16.164 example/train_dist_gat_nodeclassification.py --num-trainers 8  --num-epochs 7 --num-hidden 8 --heads 4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/mag240m/2 --dataset mag240M --eval-every 21 --feat-slope 0.000532 --adj-slope 0.025124 --fan-out 10,25 --reserved-mem 2.0"
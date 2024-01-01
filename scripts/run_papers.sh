python3.10 example/train_graphsage_nodeclassification.py --num-trainers 1  --num-epochs 7 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_graphsage_nodeclassification.py --num-trainers 2  --num-epochs 7 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_graphsage_nodeclassification.py --num-trainers 4  --num-epochs 7 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_graphsage_nodeclassification.py --num-trainers 8  --num-epochs 7 --num-hidden 32 --dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_gat_nodeclassification.py --num-trainers 1  --num-epochs 7 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_gat_nodeclassification.py --num-trainers 2  --num-epochs 7 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_gat_nodeclassification.py --num-trainers 4  --num-epochs 7 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
python3.10 example/train_gat_nodeclassification.py --num-trainers 8  --num-epochs 7 --num-hidden 8 --heads 4,4,1 --feat-dropout 0.2 --attn-dropout 0.2 --lr 0.003 --root /home/ubuntu/workspace/compressed_dataset/ogbn-papers100M/2 --dataset ogbn-papers100M --eval-every 21 --feat-slope 0.000516 --adj-slope 0.031503 --fan-out 12,12,12 --reserved-mem 2.0
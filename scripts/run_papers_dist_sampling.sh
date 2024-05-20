for i in 4 2 1
do
   python3 examples/launch_dist_train.py --workspace ~/f2cgt_workspace/F2CGT_v2 \
      --num_trainers 8 \
      --num_samplers 1 \
      --num_servers 1 \
      --part_config /home/ubuntu/dgl_workspace/partition_dataset/papers_${i}p/ogb-paper100M.json \
      --ip_config examples/ip_config${i}.txt \
      "/home/ubuntu/miniconda3/envs/raft/bin/python3 examples/train_graphsage_nodeclass_dist.py \
         --graph-name ogb-paper100M \
         --ip-config examples/ip_config${i}.txt \
         --root /home/ubuntu/f2cgt_workspace/v2_dataset/papers \
         --compress-root /home/ubuntu/f2cgt_workspace/v2_dataset/papers/sq32 \
         --num-gpus 8 \
         --num-epochs 4 --eval-every 999 \
         --model sage --num-hidden 128 --num-heads 8 \
         --fan-out 20,20,20 --batch-size 1536 \
         --fusion --create-cache --reserved-mem 2.0 \
         --breakdown"
   python3 examples/launch_dist_train.py --workspace ~/f2cgt_workspace/F2CGT_v2 \
      --num_trainers 8 \
      --num_samplers 1 \
      --num_servers 1 \
      --part_config /home/ubuntu/dgl_workspace/partition_dataset/papers_${i}p/ogb-paper100M.json \
      --ip_config examples/ip_config${i}.txt \
      "/home/ubuntu/miniconda3/envs/raft/bin/python3 examples/train_graphsage_nodeclass_dist.py \
         --graph-name ogb-paper100M \
         --ip-config examples/ip_config${i}.txt \
         --root /home/ubuntu/f2cgt_workspace/v2_dataset/papers \
         --compress-root /home/ubuntu/f2cgt_workspace/v2_dataset/papers/sq32 \
         --num-gpus 8 \
         --num-epochs 4 --eval-every 999 \
         --model gat --num-hidden 128 --num-heads 8 \
         --fan-out 20,20,20 --batch-size 1536 \
         --fusion --create-cache --reserved-mem 2.0 \
         --breakdown"
done

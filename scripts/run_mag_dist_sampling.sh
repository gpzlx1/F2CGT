for i in 4 2 1
do
   python3 examples/launch_dist_train.py --workspace ~/f2cgt_workspace/F2CGT_v2 \
      --num_trainers 8 \
      --num_samplers 1 \
      --num_servers 1 \
      --part_config /home/ubuntu/dgl_workspace/partition_dataset/mag240m_${i}p/mag240m.json \
      --ip_config examples/ip_config${i}.txt \
      "/home/ubuntu/miniconda3/envs/raft/bin/python3 examples/train_graphsage_nodeclass_dist.py \
         --graph-name mag240m \
         --ip-config examples/ip_config${i}.txt \
         --root /home/ubuntu/f2cgt_workspace/v2_dataset/mag240m \
         --compress-root /home/ubuntu/f2cgt_workspace/v2_dataset/mag240m/sq8,vq128-distsampling \
         --num-gpus 8 \
         --num-epochs 4 --eval-every 999 \
         --model sage --num-hidden 128 --num-heads 8 \
         --fan-out 5,10,15 --batch-size 1536 \
         --fusion \
         --breakdown"
   python3 examples/launch_dist_train.py --workspace ~/f2cgt_workspace/F2CGT_v2 \
      --num_trainers 8 \
      --num_samplers 1 \
      --num_servers 1 \
      --part_config /home/ubuntu/dgl_workspace/partition_dataset/mag240m_${i}p/mag240m.json \
      --ip_config examples/ip_config${i}.txt \
      "/home/ubuntu/miniconda3/envs/raft/bin/python3 examples/train_graphsage_nodeclass_dist.py \
         --graph-name mag240m \
         --ip-config examples/ip_config${i}.txt \
         --root /home/ubuntu/f2cgt_workspace/v2_dataset/mag240m \
         --compress-root /home/ubuntu/f2cgt_workspace/v2_dataset/mag240m/sq8,vq128-distsampling \
         --num-gpus 8 \
         --num-epochs 4 --eval-every 999 \
         --model gat --num-hidden 128 --num-heads 8 \
         --fan-out 5,10,15 --batch-size 1536 \
         --fusion \
         --breakdown"
done

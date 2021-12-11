CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type style --work_dir ./tmp/sag \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --encoder_feature_dim 50 --num_layer 4 \
    --agent sag_sac --frame_stack 3 --data_augs color_jitter  \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 \
    --batch_size 128 --num_train_steps 500000
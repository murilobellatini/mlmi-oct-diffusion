set MODEL_FLAGS=--image_size 64 --num_channels 128 --num_res_blocks 3  &
set DIFFUSION_FLAGS=--diffusion_steps 20 --noise_schedule linear  &
set TRAIN_FLAGS=--lr 1e-4 --batch_size 10  &
python scripts/image_train.py --data_dir ./data/raw/test/resized %MODEL_FLAGS% %DIFFUSION_FLAGS% %TRAIN_FLAGS% 
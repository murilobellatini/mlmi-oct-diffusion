set MODEL_FLAGS=--image_size 64 --num_channels 128 --num_res_blocks 3  &
set DIFFUSION_FLAGS=--diffusion_steps 100 --noise_schedule linear  &
python scripts/image_sample.py --model_path "C:\Users\muril\AppData\Local\Temp\openai-2022-06-06-21-17-10-141600\model000000.pt" %MODEL_FLAGS% %DIFFUSION_FLAGS%
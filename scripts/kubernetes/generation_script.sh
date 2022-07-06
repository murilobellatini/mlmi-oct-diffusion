sudo apt update
sudo apt-get install libopenmpi-dev -y
echo "MPICH Installed"
cd /mnt/nfs-students/mlmi-oct-diffusion
env MPICC=/usr/bin/mpicc pip install mpi4py
pip install -r requirements.txt
'PATH=$PATH:/home/celikfurkan2/.local/bin'
export OPENAI_LOGDIR="/mnt/nfs-students/mlmi-oct-diffusion/outputs"
#Don't forget to change model path
python scripts/image_sample.py --model_path="./model/model036900.pt"
--image_size=128 --num_channels=128 --num_res_blocks=3
--diffusion_steps=100 --noise_schedule=linear --num_samples=10
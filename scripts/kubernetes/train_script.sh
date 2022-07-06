sudo apt update
sudo apt-get install libopenmpi-dev -y
echo "MPICH Installed"
sudo apt install git
git --version
cd /mnt/nfs-students
#Actually, besided that since the repo is private you need to set up an ssh key and store it. Therefore, I wouldn't advise this to be used.
#git config --global user.name "your_username"
#git config --global user.email "your_email"
#You can select your branch in here instead of master
#git clone --branch master https://github.com/murilobellatini/mlmi-oct-diffusion.git
cd mlmi-oct-diffusion
pip install --upgrade setuptools
env MPICC=/usr/bin/mpicc pip install mpi4py
# PLEASE CHANGE USERNAME IN HERE TO YOUR USERNAME
'PATH=$PATH:/home/celikfurkan2/.local/bin'
pip install -r requirements.txt
mpiexec -n 2 python scripts/image_train.py params.yaml
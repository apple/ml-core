#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
apt-get update
apt install -y less tmux psmisc curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
        libosmesa6-dev software-properties-common net-tools unzip vim \
        virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf xvfb ffmpeg git

# Download distraction videos.
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip -q DAVIS-2017-trainval-480p.zip

# Install MuJoCo
mkdir ~/.mujoco

# Mujoco 2.1.0 for dm_control
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
mv mujoco210 ~/.mujoco/mujoco210

# Install MuJoCo 2.0
wget https://www.roboti.us/download/mujoco200_linux.zip
wget https://roboti.us/file/mjkey.txt
unzip mujoco200_linux.zip
mv mujoco200_linux ~/.mujoco/mujoco200_linux
mv mjkey.txt ~/.mujoco/
ln -s ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH

# Put 10_nvidia.json in the right place.
# This is needed to make the renderer use gpu.
cp 10_nvidia.json /usr/share/glvnd/egl_vendor.d/

# Mujoco-py uses a very brittle test to determine whether to use gpu rendering.
# It looks for a directory named /usr/lib/nvidia-xxx (but doesn't really need or use any libraries present there).
# So we just create a dummy one here.
mkdir -p /usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

# Create conda environment.
conda config --set remote_read_timeout_secs 600
conda env create -f conda_env_robosuite.yml

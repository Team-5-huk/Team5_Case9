sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev
pip3 install Cython==0.29.33 virtualenv  # the --user should be removed if you do this in a venv

# add the following line at the end of your ~/.bashrc file
export PATH=$PATH:~/.local/bin/

pip install "kivy[base]"


scp -P 54321 servervf@87.244.7.150:/home/servervf/case-19/mobile_app/cvapp9.apk /home/tommy/desktop

buildozer -v android debug 
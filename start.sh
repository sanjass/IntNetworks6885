conda activate int_net;
export PYTHONPATH=`pwd`;
eval "$(ssh-agent -s)";
ssh-add ~/.ssh/id_rsa;

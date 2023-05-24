python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 80 --deep_eval_freq 20
python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 50 --deep_eval_freq 20
python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar10.yaml --labeled-num 5 --deep_eval_freq 20

python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 80 --deep_eval_freq 1 --layer proj
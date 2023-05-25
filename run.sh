python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 80 --deep_eval_freq 20
python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 50 --deep_eval_freq 20
python run_ncd.py --config-file configs/supspectral_resnet_mlp1000_norelu_cifar10.yaml --labeled-num 5 --deep_eval_freq 20

python run_ncd.py --config-file configs/spectral_resnet_mlp1000_norelu_cifar100_lr003_mu1.yaml --labeled-num 80 --deep_eval_freq 20 --layer penul
python run_ncd.py --config-file configs/spectral_resnet_mlp1000_norelu_cifar100_lr003_mu1.yaml --labeled-num 50 --deep_eval_freq 20 --layer penul
python run_ncd.py --config-file configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml --labeled-num 5 --deep_eval_freq 20 --layer penul

python run_unsup.py --config-file configs/spectral_resnet_mlp1000_norelu_cifar100_lr003_mu1.yaml --labeled-num 80 --deep_eval_freq 20 --layer penul
python run_unsup.py --config-file configs/spectral_resnet_mlp1000_norelu_cifar100_lr003_mu1.yaml --labeled-num 50 --deep_eval_freq 20 --layer penul
python run_unsup.py --config-file configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml --labeled-num 5 --deep_eval_freq 20 --layer penul


CUDA_VISIBLE_DEVICES=0 python run_ncd.py --data_dir ../supspec/data --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 80 --deep_eval_freq 20 --layer penul

cd ../NSCL/
CUDA_VISIBLE_DEVICES=1 python run_ncd.py --data_dir ../supspec/data --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 50 --deep_eval_freq 20 --layer penul

cd ../NSCL/
CUDA_VISIBLE_DEVICES=2 python run_ncd.py --data_dir ../supspec/data --config-file configs/supspectral_resnet_mlp1000_norelu_cifar10.yaml --labeled-num 5 --deep_eval_freq 20 --layer penul

cd ../NSCL/
CUDA_VISIBLE_DEVICES=3 python run_ncd.py --data_dir ../supspec/data --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 80 --deep_eval_freq 20 --layer proj

cd ../NSCL/
CUDA_VISIBLE_DEVICES=4 python run_ncd.py --data_dir ../supspec/data --config-file configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml --labeled-num 50 --deep_eval_freq 20 --layer proj

cd ../NSCL/
CUDA_VISIBLE_DEVICES=5 python run_ncd.py --data_dir ../supspec/data --config-file configs/supspectral_resnet_mlp1000_norelu_cifar10.yaml --labeled-num 5 --deep_eval_freq 20 --layer proj



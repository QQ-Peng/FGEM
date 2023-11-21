# Requirements
python==3.7
torch==1.13.0+cu116
torch-geometric==2.1.0

# Run model on benchmark datasets
python train.py --dataset {cora,citeseer,pubmed} --G_prior {sbm,lsm} --lc {20,10} --er {1.0,0.5}  --device {cpu,cuda}

lc: labels number per class for training
er: edges ratio for training.

# Run model on PPI dataset
python train.py --dataset multiPPI --G_prior {sbm,lsm}  --device {cpu,cuda}

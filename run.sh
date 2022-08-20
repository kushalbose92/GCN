python -u main.py --dataset 'cora' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.50 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 



python -u main.py --dataset 'citeseer' --lr 0.005 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.50 --w_decay 0.0005 --device 'cuda:0' | tee output.txt 



python -u main.py --dataset 'pubmed' --lr 0.01 --seed 0 --num_layers 2 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --nheads 1 --alpha 0.2 --dropout 0.10 --w_decay 0.001 --device 'cuda:0' | tee output.txt 


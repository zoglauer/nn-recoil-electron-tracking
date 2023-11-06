model_iter=modelv25_directed_lr0001_batchsize4_hidden60
mkdir $model_iter
python3 train.py --lr .0001 --batch-size 4 --hidden-size 60 --epochs 200 --model-iteration $model_iter >> $model_iter.txt

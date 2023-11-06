model_iter=modelv1
mkdir $model_iter
python3 train.py --lr .0001 --batch-size 4 --epochs 100 --model-iteration $model_iter >> $model_iter.txt

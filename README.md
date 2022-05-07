# 3E-Solver-IJCAI

This reposity is official implementation of "3E-Slover: An Effortless, Easy-to-Update, and End-to-End Solver with Semi-Supervised Learning for Breaking Text-Based Captchas" (IJCAI-2022).

## Dependency
```
torch=1.9.0+cu111
torchvision=0.10.0+cu111
numpy=1.21.2
PIL=8.3.1
matplotlib=3.4.3
```

## Code
``` shell
python FixMatch_2.py --dataset apple --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset ganji-1 --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset google --label 500.txt --unlabeled-number 5000
python FixMatch_2.py --dataset microsoft --label 500.txt --unlabeled-number 5000
python FixMatch_2.py --dataset sina --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset weibo --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset wikipedia --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
python FixMatch_2.py --dataset yandex --label 500.txt --unlabeled-number 5000 --epoch 400 --lr 0.01
```

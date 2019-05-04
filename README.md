# bayesian_regression
ベイズ線形回帰のデモコード

## requirement  
```
numpy
matplotlib
argparse  
```

## usage
```
$git clone https://github.com/wildgeece96/bayesian_regression  
```
ガウス基底関数を使う場合
```
$python3 main.py --M 40 --mode gauss
```

多項式を使う場合  
```
$python3 main.py --M 6 --mode polynommial
```

結果は`fig`フォルダ内に作られます。  

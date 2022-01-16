## Running the code

#### Train

```
python eszsl/eszsl_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -mode train -alpha [KERNEL SPACE REGULARIZER] -gamma [ATT SPACE REGULARIZER] -acc [ACCURACY TYPE]
```
For testing, set mode to test and set alpha, gamma to best combination from tables below.

## Results

The numbers below are **class-averaged top-1 accuracies**.

#### Classical ZSL

| Dataset | ZSLGBU Results| Repository Results | Hyperparams from Val |
|---------|:-------------:|:------------------:|:--------------------:|
| CUB     |     53.9      | 	   53.94 	   |Alpha=3, Gamma=-1     |
| AWA1    |   **58.2**    |        56.80       |Alpha=3, Gamma=0      |
| AWA2    |   **58.6**    |        54.82       |Alpha=3, Gamma=0      |
| aPY     |     38.3      |      **38.56**     |Alpha=3, Gamma=-1     |
| SUN     |     54.5      |      **55.69**     |Alpha=3, Gamma=2      |
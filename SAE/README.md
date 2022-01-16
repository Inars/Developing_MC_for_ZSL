## Running the code

#### Train

```
python sae/sae_gzsl.py -data AWA2/AWA1/CUB/SUN/APY -mode train -ld1 [LOWER BOUND OF VARIATION] -ld2 [UPPER BOUND OF VARIATION] -acc [ACCURACY TYPE]
```
For testing, set mode to test and set ld1 (F->S) and ld2 (S->F) to the best values from the tables below.

## Results

The numbers below are **class-averaged top-1 accuracies**.

#### Classical ZSL

| Dataset | ZSLGBU Results || Repository Results                    |||
|---------|:--------------:|:--------:|:------:|:----------:|:-------:|
|         |                | F->S (W) | Lambda | S->F (W.T) | Lambda  |
| CUB     | 33.3           | 39.48    | 100    | **46.70**  | 0.2     |
| AWA1    | 53.0           | 51.34    | 3.0    | **59.89**  | 0.8     |
| AWA2    | 54.1           | 51.66    | 0.6    | **60.51**  | 0.2     |
| aPY     | 8.3            | 16.07    | 2.0    | **16.50**  | 4.0     |
| SUN     | 40.3           | 52.85    | 0.32   | **59.86**  | 0.16    |
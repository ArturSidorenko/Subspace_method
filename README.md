# Subspace_method
A directory for a subspace method. It was used for a paper published in JSAA 
https://ijassa.ipu.ru/index.php/ijassa/article/view/1492

## StateSpaceModel

A linear state-space model is described as a time series:

$$
x_{t+1}=Ax_t,\qquad y_t=Cx_t,
$$

where $A$ is the transition matrix, $C$ is the observation matrix, and `k` is the dimension (meta parameter).

The model is based on a block Hankel matrix and truncated SVD.

## Usage

`X` is numeric pandas DataFrame or NumPy array `(n_samples, n_features)`, rows in chronological order.

```python
model = StateSpaceModel(k=4).fit(X)

restored = model.filtrate()       # reconstruct the training series
forecast = model.predict(10)     # forecast the next 10 steps
periods = model.periods()        # estimate periods in sampling steps
```

We use `numpy`, `pandas`, `warnings`. We can also use `fit` method.

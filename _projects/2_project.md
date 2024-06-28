---
layout: page
title: Lab 2
description: Second lab assignment for the Causal AI course
img: assets/img/3.jpg
importance: 2
category: Causal AI course
giscus_comments: false
---


# Potential Outcomes and RCTs

#### Group 3: Dube, V., Garay, E. Guerrero, J., Villalba, M.

## Multicollinearity

Multicolinearity occurs when two or more predictors in a regresion model are highly correlated to one another, causing a higher variance our the estimated coefficients. To understand the way multicollinearity affects our regresion we can examine the composition of the variance of our estimates. 

Suppose some Data Generating Process follows:

$$
\begin{equation*}
  Y = X_1\beta_1 +\epsilon
\end{equation*}
$$


Considering the partitioned regression model:

$$
\begin{align*}
  Y &= X\beta + e \\
  Y &= X_1\beta_1 + X_2\beta_2 + e
\end{align*}
$$

We know that the OLS estimator will solve this equation:

$$
\begin{align*}
(X'X)\hat{\beta} &=X'Y \\

\begin{bmatrix}
  X_1'X_1      & X_1'X_2   \\
  X_2'X_1      & X_2'X_2 
\end{bmatrix}
\begin{bmatrix}
\hat{\beta_1} \\
\hat{\beta_2}
\end{bmatrix}
& =
\begin{bmatrix}
X_1'Y \\
X_2'Y
\end{bmatrix}
\end{align*}
$$

This, because of the Frisch-Whaugh-Lovell Theorem, yields:

$$
\begin{align*}
  \hat{\beta_1} &= (X_1'M_2X_1)^{-1}X_1'M_2Y
\end{align*}
$$

Where $$M_2 = I - X_2(X_2'X_2)^{-1}X_2'$$, is the orthogonal projection matrix to $$X_2$$.

Note that $$M_2$$ is symmetric, idempotent, and that any variable premultiplied by it yields the residual from from running $$X_2$$ on that variable. For an arbitrary variable $$Z$$:

$$
\begin{align*}
  M_2Z &= (I - X_2(X_2'X_2)^{-1}X_2')Z \\
  &= Z - X_2(X_2'X_2)^{-1}X_2'Z \\
  &= Z - X_2\hat{\omega} \\
  &= Z - \hat{Z} \\
  &= e_{Z}
\end{align*}
$$

Where $$e_{Z}$$ and $$\hat{\omega} \equiv (X_2'X_2)^{-1}X_2'Z$$ come from the regresion: $$ Z = X_2\hat{\omega} + e_{Z}$$

In a sense, the $$M_2$$ matrix cleanses or "filters out" our $$Z$$ variable, keeping only the part which is orthogonal to $$X_2$$.

Returning to $$\hat{\beta_1}$$:

$$
\begin{align*}
  \hat{\beta_1} &= (X_1'M_2X_1)^{-1}X_1'M_2Y \\
  &= (X_1'M_2X_1)^{-1}X_1'M_2(X_1\beta_1 + \epsilon) \\
  &= \beta_1 + (X_1'M_2X_1)^{-1}X_1'M_2\epsilon 
\end{align*}
$$

For the conditional variance of $$\hat{\beta_1}$$ this has great implications:

$$
\begin{align*}
  Var(\hat{\beta_1}|X) &= Var(\beta_1 + (X_1'M_2X_1)^{-1}X_1'M_2\epsilon|X) \\
  &= Var((X_1'M_2X_1)^{-1}X_1'M_2\epsilon|X) \\
  &= E[((X_1'M_2X_1)^{-1}X_1'M_2\epsilon)((X_1'M_2X_1)^{-1}X_1'M_2\epsilon)'|X] \\
  &= E[(X_1'M_2X_1)^{-1}X_1'M_2\epsilon\epsilon'M_2'X_1(X_1'M_2'X_1)^{-1}|X] \\
  &= (X_1'M_2X_1)^{-1}X_1'M_2E[\epsilon\epsilon'|X]M_2'X_1(X_1'M_2'X_1)^{-1}
\end{align*}
$$

Under the traditional assumption that $$E[\epsilon\epsilon'|X] = \sigma^2I$$:

$$
\begin{align*}
 Var(\hat{\beta_1}|X) &= \sigma^2(X_1'M_2X_1)^{-1}X_1'M_2M_2'X_1(X_1'M_2'X_1)^{-1} \\
&= \sigma^2(X_1'M_2'X_1)^{-1}
\end{align*}
$$

Remembering that the variance of $$X_1$$ can be decomposed into two positive components:

$$
\begin{align*}
  X_1 &= X_2\alpha + v \\
  Var(X_1) &= Var(X_2\alpha) + Var(v) \\
  Var(X_1) - Var(X_2\alpha) &= Var(v) \\
  E[X_1'X_1] - Var(X_2\alpha) &= E[X_1'M_2'X_1]
\end{align*}
$$

Thus, necessarily: $$E[X_1'M_2X_1] \leq E[X_1'X_1]$$ 

Altogether this means: $$\sigma^2(X_1'X_1)^{-1} \leq \sigma^2(X_1'M_2'X_1)^{-1}$$

This shows that controlling for the irrelevant variables $$X_2$$ will in fact increase the variance of $$\hat{\beta_1}$$ by limiting us to the "usable" variance of $$X_1$$ which is orthogonal to $$X_2$$. 

Suppose we want to estimate the impact of years of schooling on future salary. Imagine as well that we have a vast array of possible control variables at our disposal. Someone who is not familiar with the concept of multicollinearity might think that to avoid any possibility of ommited variable bias and ensure consistency it is best to control for everything we can. We now know this is not the case and that this approach can inadvertently introduce multicollinearity. 

Consider that we have as a possible control variable the total number of courses taken by each student. Intuitively, years of schooling are likely to correlate strongly with the number of total courses taken (more years in school tipically leads to more courses completed) and so controlling for this variable may result in the problem described above, inflating the variance of the estimated coefficients and potentially distorting our understanding of the true effect of schooling on salary. 

The same rationale applies to many other examples. For instance, imagine estimating the impact of marketing expenditure on sales. Controlling for variable such as number of marketing campaigns will probably cause the same issue.


### Perfectly collinear regressors

A special case of the previously mentioned concept of multicollinearity arises when a variable is a linear combination of some other variables from our dataset, so not only are these variables highly correlated, but we say that they are perfectly collinear.

Considering the partitioned regression model:

$$
\begin{align*}
    Y &= x_1\beta_1 + X_2\beta_2 + \epsilon \\
    x_1 &= X_2 \alpha
\end{align*}
$$

where the second equation is deterministic.

We know that the OLS estimator will solve this equation:

$$
\begin{align*}
(X'X)\hat{\beta} &=X'Y \\

\begin{bmatrix}
  X_1'X_1      & X_1'X_2   \\
  X_2'X_1      & X_2'X_2 
\end{bmatrix}
\begin{bmatrix}
\hat{\beta_1} \\
\hat{\beta_2}
\end{bmatrix}
& =
\begin{bmatrix}
X_1'Y \\
X_2'Y
\end{bmatrix}
\end{align*}
$$

Substituting $$X_1 = X_2 \alpha$$ on the $$(X'X)$$ matrix:

$$
\begin{align*}
(X'X) &=
\begin{bmatrix}
  (X_2\alpha)'X_2\alpha      & (X_2\alpha)'X_2   \\
  X_2'(X_2\alpha)      & X_2'X_2 
\end{bmatrix} \\

&=
\begin{bmatrix}
  \alpha'X_2'X_2\alpha      & \alpha'X_2'X_2   \\
  X_2'X_2\alpha      & X_2'X_2 
\end{bmatrix}
\end{align*}
$$

This yields the determinant:

$$
\begin{align*}
\det (X'X) &= \det\left(
\begin{bmatrix}
  \alpha'X_2'X_2\alpha      & \alpha'X_2'X_2   \\
  X_2'X_2\alpha      & X_2'X_2 
\end{bmatrix} \right) \\
\end{align*}
$$

Transforming the matrix using row operations:

$$
\begin{align*}
\det (X'X) &= \det\left(
\begin{bmatrix}
  I      & -\alpha'   \\
  0      & I 
\end{bmatrix}
\begin{bmatrix}
  \alpha'X_2'X_2\alpha      & \alpha'X_2'X_2   \\
  X_2'X_2\alpha      & X_2'X_2 
\end{bmatrix} \right) \\
&= \det\left(
\begin{bmatrix}
  0      & 0   \\
  X_2'X_2\alpha      & X_2'X_2 
\end{bmatrix} \right)
\end{align*}
$$

Like this, we can see that:

$$
\begin{align*}
\det (X'X) &= 0
\end{align*}
$$

Because of this, $$(X'X)$$ is not invertible and there is no solution for the OLS estimation



### Practical application

We can easily show what we have theoretically explained with a practical application. We will create a dataset simulating the regressors that we may want to include in the estimation of a linear model.

The first 9 variables follow a normal distribution.


```python
import numpy as np
import pandas as pd
from numpy.linalg import inv, LinAlgError

# Seed for reproducibility
np.random.seed(0)

# Generate 9 vectors of normal distributions
X = np.random.randn(10, 9)
pd.DataFrame(X)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>0.400157</td>
      <td>0.978738</td>
      <td>2.240893</td>
      <td>1.867558</td>
      <td>-0.977278</td>
      <td>0.950088</td>
      <td>-0.151357</td>
      <td>-0.103219</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.410599</td>
      <td>0.144044</td>
      <td>1.454274</td>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
      <td>0.653619</td>
      <td>0.864436</td>
      <td>-0.742165</td>
      <td>2.269755</td>
      <td>-1.454366</td>
      <td>0.045759</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.187184</td>
      <td>1.532779</td>
      <td>1.469359</td>
      <td>0.154947</td>
      <td>0.378163</td>
      <td>-0.887786</td>
      <td>-1.980796</td>
      <td>-0.347912</td>
      <td>0.156349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.230291</td>
      <td>1.202380</td>
      <td>-0.387327</td>
      <td>-0.302303</td>
      <td>-1.048553</td>
      <td>-1.420018</td>
      <td>-1.706270</td>
      <td>1.950775</td>
      <td>-0.509652</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.438074</td>
      <td>-1.252795</td>
      <td>0.777490</td>
      <td>-1.613898</td>
      <td>-0.212740</td>
      <td>-0.895467</td>
      <td>0.386902</td>
      <td>-0.510805</td>
      <td>-1.180632</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.028182</td>
      <td>0.428332</td>
      <td>0.066517</td>
      <td>0.302472</td>
      <td>-0.634322</td>
      <td>-0.362741</td>
      <td>-0.672460</td>
      <td>-0.359553</td>
      <td>-0.813146</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.726283</td>
      <td>0.177426</td>
      <td>-0.401781</td>
      <td>-1.630198</td>
      <td>0.462782</td>
      <td>-0.907298</td>
      <td>0.051945</td>
      <td>0.729091</td>
      <td>0.128983</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.139401</td>
      <td>-1.234826</td>
      <td>0.402342</td>
      <td>-0.684810</td>
      <td>-0.870797</td>
      <td>-0.578850</td>
      <td>-0.311553</td>
      <td>0.056165</td>
      <td>-1.165150</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.900826</td>
      <td>0.465662</td>
      <td>-1.536244</td>
      <td>1.488252</td>
      <td>1.895889</td>
      <td>1.178780</td>
      <td>-0.179925</td>
      <td>-1.070753</td>
      <td>1.054452</td>
    </tr>
  </tbody>
</table>
</div>



However, the 10th variable is a linear combination (the sum) of variables 1, 5 and 9.


```python
# Create the 10th vector as a linear combination of vectors 1, 5, and 9
a, b, c = 1, 1, 1
X = np.hstack([X, (a*X[:, 0] + b*X[:, 4] + c*X[:, 8]).reshape(10,1)])
pd.DataFrame(X)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.764052</td>
      <td>0.400157</td>
      <td>0.978738</td>
      <td>2.240893</td>
      <td>1.867558</td>
      <td>-0.977278</td>
      <td>0.950088</td>
      <td>-0.151357</td>
      <td>-0.103219</td>
      <td>3.528391</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.410599</td>
      <td>0.144044</td>
      <td>1.454274</td>
      <td>0.761038</td>
      <td>0.121675</td>
      <td>0.443863</td>
      <td>0.333674</td>
      <td>1.494079</td>
      <td>-0.205158</td>
      <td>0.327115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.313068</td>
      <td>-0.854096</td>
      <td>-2.552990</td>
      <td>0.653619</td>
      <td>0.864436</td>
      <td>-0.742165</td>
      <td>2.269755</td>
      <td>-1.454366</td>
      <td>0.045759</td>
      <td>1.223262</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.187184</td>
      <td>1.532779</td>
      <td>1.469359</td>
      <td>0.154947</td>
      <td>0.378163</td>
      <td>-0.887786</td>
      <td>-1.980796</td>
      <td>-0.347912</td>
      <td>0.156349</td>
      <td>0.347328</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.230291</td>
      <td>1.202380</td>
      <td>-0.387327</td>
      <td>-0.302303</td>
      <td>-1.048553</td>
      <td>-1.420018</td>
      <td>-1.706270</td>
      <td>1.950775</td>
      <td>-0.509652</td>
      <td>-0.327914</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.438074</td>
      <td>-1.252795</td>
      <td>0.777490</td>
      <td>-1.613898</td>
      <td>-0.212740</td>
      <td>-0.895467</td>
      <td>0.386902</td>
      <td>-0.510805</td>
      <td>-1.180632</td>
      <td>-1.831447</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.028182</td>
      <td>0.428332</td>
      <td>0.066517</td>
      <td>0.302472</td>
      <td>-0.634322</td>
      <td>-0.362741</td>
      <td>-0.672460</td>
      <td>-0.359553</td>
      <td>-0.813146</td>
      <td>-1.475651</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.726283</td>
      <td>0.177426</td>
      <td>-0.401781</td>
      <td>-1.630198</td>
      <td>0.462782</td>
      <td>-0.907298</td>
      <td>0.051945</td>
      <td>0.729091</td>
      <td>0.128983</td>
      <td>-1.134517</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.139401</td>
      <td>-1.234826</td>
      <td>0.402342</td>
      <td>-0.684810</td>
      <td>-0.870797</td>
      <td>-0.578850</td>
      <td>-0.311553</td>
      <td>0.056165</td>
      <td>-1.165150</td>
      <td>-0.896546</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.900826</td>
      <td>0.465662</td>
      <td>-1.536244</td>
      <td>1.488252</td>
      <td>1.895889</td>
      <td>1.178780</td>
      <td>-0.179925</td>
      <td>-1.070753</td>
      <td>1.054452</td>
      <td>3.851167</td>
    </tr>
  </tbody>
</table>
</div>



As we saw in theory, this should cause our dataset to have a determinant of zero. Thus, making X singular and yielding an error message when trying to invert it.


```python
np.linalg.det(X)
```




    1.0098307832528084e-13




```python
pd.DataFrame(inv(X))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.199889e+15</td>
      <td>1.801440e+15</td>
      <td>1.392735e+15</td>
      <td>1.415658e+15</td>
      <td>9.653012e+13</td>
      <td>3.252962e+14</td>
      <td>-5.392287e+14</td>
      <td>-4.107423e+14</td>
      <td>2.334001e+14</td>
      <td>2.658898e+14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.568882e-01</td>
      <td>-1.607297e+00</td>
      <td>-1.452389e+00</td>
      <td>-1.581686e+00</td>
      <td>1.002832e+00</td>
      <td>1.810799e+00</td>
      <td>2.414320e-01</td>
      <td>-9.804048e-01</td>
      <td>-2.591498e+00</td>
      <td>8.121699e-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.729573e-01</td>
      <td>-3.174262e-01</td>
      <td>-3.248933e-01</td>
      <td>-1.024257e-01</td>
      <td>-4.607751e-02</td>
      <td>1.494138e-01</td>
      <td>-1.582449e-01</td>
      <td>-2.141899e-01</td>
      <td>-2.589349e-01</td>
      <td>-2.381031e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.314975e+00</td>
      <td>1.663818e+00</td>
      <td>1.316748e+00</td>
      <td>1.302163e+00</td>
      <td>-7.972630e-01</td>
      <td>-1.528744e+00</td>
      <td>3.777566e-02</td>
      <td>5.674403e-01</td>
      <td>1.639215e+00</td>
      <td>-3.485951e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.199889e+15</td>
      <td>1.801440e+15</td>
      <td>1.392735e+15</td>
      <td>1.415658e+15</td>
      <td>9.653012e+13</td>
      <td>3.252962e+14</td>
      <td>-5.392287e+14</td>
      <td>-4.107423e+14</td>
      <td>2.334001e+14</td>
      <td>2.658898e+14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.802669e-01</td>
      <td>-5.811803e-01</td>
      <td>-8.573224e-01</td>
      <td>-9.797333e-01</td>
      <td>2.685130e-01</td>
      <td>7.978812e-01</td>
      <td>2.750149e-01</td>
      <td>-4.043804e-01</td>
      <td>-9.763545e-01</td>
      <td>2.882150e-01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.085413e-01</td>
      <td>-1.040756e+00</td>
      <td>-7.763937e-01</td>
      <td>-1.195168e+00</td>
      <td>4.913696e-01</td>
      <td>1.007714e+00</td>
      <td>-5.571324e-02</td>
      <td>-6.134050e-01</td>
      <td>-1.578493e+00</td>
      <td>-2.636936e-01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-4.802377e-01</td>
      <td>1.078846e+00</td>
      <td>5.833448e-01</td>
      <td>5.360183e-01</td>
      <td>-6.271024e-02</td>
      <td>-4.305357e-01</td>
      <td>-1.356475e-01</td>
      <td>3.729121e-01</td>
      <td>6.578932e-01</td>
      <td>1.184780e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.199889e+15</td>
      <td>1.801440e+15</td>
      <td>1.392735e+15</td>
      <td>1.415658e+15</td>
      <td>9.653012e+13</td>
      <td>3.252962e+14</td>
      <td>-5.392287e+14</td>
      <td>-4.107423e+14</td>
      <td>2.334001e+14</td>
      <td>2.658898e+14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.199889e+15</td>
      <td>-1.801440e+15</td>
      <td>-1.392735e+15</td>
      <td>-1.415658e+15</td>
      <td>-9.653012e+13</td>
      <td>-3.252962e+14</td>
      <td>5.392287e+14</td>
      <td>4.107423e+14</td>
      <td>-2.334001e+14</td>
      <td>-2.658898e+14</td>
    </tr>
  </tbody>
</table>
</div>



We can see that this is not the case. Is this a contradiction to our theoretical proof? Python, as well as Julia, yield a determinant extremely close to cero but not equal to cero, so they are able to find a "supposed" inverse to the matrix. This would seem to contradict what we have proven in theory, however, this is a problem rooted in the way those languages handle float values. Thus, that is not a contradiction of theory but rather an illustration of how computational environments deal differently with the inherent limitations of floating-point arithmetic.

## Analyzing RCT data with precision adjustment


```python
pip install pandas matplotlib seaborn statsmodels scikit-learn
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data

Penn = pd.read_csv("C:/Users/Erzo/Documents/GitHub/CausalAI-Course/data/penn_jae.dat" , sep='\s', engine='python')
#Penn = pd.read_csv("../../../data/penn_jae.dat" , sep='\s', engine='python')
# Filtering to focus on Treatment Group 2 and Control Group
Penn_filtered = Penn[(Penn['tg'] == 2) | (Penn['tg'] == 0)]
# Actualiza los valores de 'tg' a 1 donde 'tg' es igual a 2
Penn_filtered.update(Penn_filtered[Penn_filtered['tg'] == 2][['tg']].replace(to_replace=2, value=1))

# Plotting histograms for the outcome variable 'inuidur1'
plt.figure(figsize=(12, 6))
sns.histplot(Penn_filtered[Penn_filtered['tg'] == 1]['inuidur1'], color='blue', label='Treatment Group 2', kde=True)
sns.histplot(Penn_filtered[Penn_filtered['tg'] == 0]['inuidur1'], color='red', label='Control Group', kde=True)
plt.title('Distribution of inuidur1 for Treatment Group 2 and Control Group')
plt.xlabel('inuidur1')
plt.ylabel('Frequency')
plt.legend()
plt.show()

```

    C:\Users\Erzo\AppData\Local\Temp\ipykernel_25196\2367704136.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Penn_filtered.update(Penn_filtered[Penn_filtered['tg'] == 2][['tg']].replace(to_replace=2, value=1))
    


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p2_output_15_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    



```python
#1 Classical 2-Sample Approach (CL)
import statsmodels.api as sm

# Filtering treatment and control groups
treatment = Penn_filtered[Penn_filtered['tg'] == 1]['inuidur1']
control = Penn_filtered[Penn_filtered['tg'] == 0]['inuidur1']

# Calculate the mean difference using Independent T-test
t2_sample = sm.stats.ttest_ind(treatment, control)
print(f"Classical 2-Sample Approach t-statistic: {t2_sample[0]}, p-value: {t2_sample[1]}")
```

    Classical 2-Sample Approach t-statistic: -2.5833616019006493, p-value: 0.009808603470207428
    


```python
#2 Classical Linear Regression Adjustment (CRA)

cra_model = sm.OLS.from_formula('inuidur1 ~ tg+ (female+black+othrace+C(dep)+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)**2', data=Penn_filtered)
cra_results = cra_model.fit()
print(cra_results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               inuidur1   R-squared:                       0.057
    Model:                            OLS   Adj. R-squared:                  0.039
    Method:                 Least Squares   F-statistic:                     3.139
    Date:                Wed, 24 Apr 2024   Prob (F-statistic):           4.49e-25
    Time:                        00:07:59   Log-Likelihood:                -21649.
    No. Observations:                5782   AIC:                         4.352e+04
    Df Residuals:                    5671   BIC:                         4.426e+04
    Df Model:                         110                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              10.3939      0.681     15.254      0.000       9.058      11.730
    C(dep)[T.1]             1.7019      1.183      1.438      0.150      -0.618       4.022
    C(dep)[T.2]            -0.1260      1.030     -0.122      0.903      -2.146       1.894
    tg                     -0.6293      0.278     -2.265      0.024      -1.174      -0.085
    female                  1.2268      0.740      1.658      0.097      -0.224       2.677
    female:C(dep)[T.1]     -1.0560      0.945     -1.118      0.264      -2.908       0.796
    female:C(dep)[T.2]      0.9122      0.836      1.091      0.275      -0.726       2.551
    black                  -0.2104      1.408     -0.149      0.881      -2.970       2.549
    black:C(dep)[T.1]       0.4702      1.434      0.328      0.743      -2.340       3.281
    black:C(dep)[T.2]      -0.3623      1.334     -0.272      0.786      -2.977       2.253
    othrace                -4.1863      5.717     -0.732      0.464     -15.395       7.022
    othrace:C(dep)[T.1]     4.9637      7.274      0.682      0.495      -9.297      19.224
    othrace:C(dep)[T.2]    -4.6474      5.746     -0.809      0.419     -15.912       6.618
    q2                      2.0576      0.925      2.225      0.026       0.245       3.870
    C(dep)[T.1]:q2         -0.5157      1.295     -0.398      0.691      -3.055       2.023
    C(dep)[T.2]:q2          2.5258      1.121      2.254      0.024       0.329       4.723
    q3                     -0.1384      0.862     -0.161      0.872      -1.828       1.551
    C(dep)[T.1]:q3         -1.3782      1.221     -1.128      0.259      -3.773       1.016
    C(dep)[T.2]:q3          1.5155      1.091      1.389      0.165      -0.624       3.655
    q4                      0.5933      0.877      0.677      0.498      -1.125       2.312
    C(dep)[T.1]:q4         -1.0021      1.216     -0.824      0.410      -3.386       1.381
    C(dep)[T.2]:q4          2.0757      1.137      1.826      0.068      -0.153       4.305
    q5                      0.8550      1.289      0.663      0.507      -1.671       3.381
    C(dep)[T.1]:q5          0.1083      1.958      0.055      0.956      -3.729       3.946
    C(dep)[T.2]:q5          3.1666      1.588      1.995      0.046       0.054       6.279
    q6                      0.3022      1.708      0.177      0.860      -3.047       3.651
    C(dep)[T.1]:q6         -0.7759      1.449     -0.536      0.592      -3.616       2.064
    C(dep)[T.2]:q6         -0.6423      1.257     -0.511      0.609      -3.107       1.822
    agelt35                 4.4301      1.250      3.545      0.000       1.980       6.880
    C(dep)[T.1]:agelt35    -1.3537      1.296     -1.045      0.296      -3.894       1.187
    C(dep)[T.2]:agelt35     1.8855      3.471      0.543      0.587      -4.919       8.690
    agegt54                 1.2482      0.982      1.271      0.204      -0.677       3.174
    C(dep)[T.1]:agegt54     1.9485      1.203      1.620      0.105      -0.409       4.306
    C(dep)[T.2]:agegt54     0.7760      1.097      0.707      0.479      -1.374       2.926
    durable                -0.0879      1.266     -0.069      0.945      -2.571       2.395
    C(dep)[T.1]:durable    -1.7719      1.487     -1.192      0.233      -4.686       1.143
    C(dep)[T.2]:durable    -1.2634      1.229     -1.028      0.304      -3.674       1.147
    lusd                    1.5218      0.877      1.735      0.083      -0.198       3.242
    C(dep)[T.1]:lusd        1.7343      1.166      1.488      0.137      -0.551       4.019
    C(dep)[T.2]:lusd        0.5890      1.044      0.564      0.573      -1.458       2.636
    husd                    3.1827      0.775      4.106      0.000       1.663       4.702
    C(dep)[T.1]:husd        1.4010      1.028      1.362      0.173      -0.615       3.417
    C(dep)[T.2]:husd       -0.9901      0.902     -1.097      0.273      -2.759       0.779
    female:black           -3.2874      0.936     -3.512      0.000      -5.122      -1.453
    female:othrace          2.0843      4.847      0.430      0.667      -7.418      11.586
    female:q2              -0.4491      0.856     -0.525      0.600      -2.127       1.229
    female:q3               2.6805      0.805      3.329      0.001       1.102       4.259
    female:q4               0.9724      0.812      1.198      0.231      -0.619       2.564
    female:q5               0.7126      1.293      0.551      0.582      -1.823       3.248
    female:q6              -1.0686      1.030     -1.038      0.299      -3.087       0.950
    female:agelt35          0.7110      1.004      0.708      0.479      -1.257       2.679
    female:agegt54          1.0233      0.916      1.117      0.264      -0.773       2.820
    female:durable         -1.4088      0.910     -1.548      0.122      -3.193       0.375
    female:lusd            -0.9064      0.791     -1.146      0.252      -2.457       0.644
    female:husd            -0.1468      0.673     -0.218      0.827      -1.466       1.172
    black:othrace       -8.661e-16   1.88e-14     -0.046      0.963   -3.77e-14     3.6e-14
    black:q2               -1.7013      1.334     -1.275      0.202      -4.317       0.915
    black:q3               -0.9967      1.219     -0.817      0.414      -3.387       1.394
    black:q4               -0.1479      1.285     -0.115      0.908      -2.667       2.371
    black:q5               -3.4132      1.897     -1.799      0.072      -7.133       0.306
    black:q6               -0.2056      2.086     -0.099      0.921      -4.295       3.884
    black:agelt35          -0.0882      1.786     -0.049      0.961      -3.590       3.413
    black:agegt54           2.0368      1.448      1.407      0.159      -0.801       4.875
    black:durable          -1.1118      1.580     -0.704      0.482      -4.209       1.986
    black:lusd              4.4008      4.124      1.067      0.286      -3.683      12.485
    black:husd             -0.2438      1.118     -0.218      0.827      -2.435       1.947
    othrace:q2              4.6462      8.753      0.531      0.596     -12.512      21.805
    othrace:q3             -6.3012      6.826     -0.923      0.356     -19.683       7.081
    othrace:q4             -0.0933      6.025     -0.015      0.988     -11.904      11.717
    othrace:q5             -6.9905      6.465     -1.081      0.280     -19.664       5.683
    othrace:q6           4.426e-15   1.25e-14      0.354      0.723   -2.01e-14    2.89e-14
    othrace:agelt35       -13.3594     14.720     -0.908      0.364     -42.217      15.498
    othrace:agegt54        14.2396      5.798      2.456      0.014       2.873      25.606
    othrace:durable        -0.5249      5.765     -0.091      0.927     -11.826      10.776
    othrace:lusd           12.0061     18.788      0.639      0.523     -24.825      48.837
    othrace:husd            1.2123      4.580      0.265      0.791      -7.766      10.191
    q2:q3                8.719e-15   2.06e-14      0.422      0.673   -3.17e-14    4.92e-14
    q2:q4               -4.373e-15   4.37e-15     -1.002      0.317   -1.29e-14    4.19e-15
    q2:q5               -2.634e-15   5.69e-15     -0.463      0.643   -1.38e-14    8.52e-15
    q2:q6                   1.4134      1.606      0.880      0.379      -1.735       4.562
    q2:agelt35             -1.2265      1.382     -0.888      0.375      -3.935       1.482
    q2:agegt54             -3.1687      1.157     -2.739      0.006      -5.436      -0.901
    q2:durable             -0.0596      1.334     -0.045      0.964      -2.675       2.556
    q2:lusd                -1.3115      1.114     -1.177      0.239      -3.496       0.873
    q2:husd                -1.0829      0.970     -1.117      0.264      -2.984       0.818
    q3:q4                1.681e-15   6.05e-15      0.278      0.781   -1.02e-14    1.35e-14
    q3:q5               -1.511e-15   5.68e-15     -0.266      0.790   -1.26e-14    9.62e-15
    q3:q6                  -0.1703      1.655     -0.103      0.918      -3.416       3.075
    q3:agelt35              2.5832      1.267      2.039      0.042       0.099       5.067
    q3:agegt54              0.0282      1.105      0.026      0.980      -2.139       2.195
    q3:durable              1.2072      1.274      0.948      0.343      -1.290       3.704
    q3:lusd                -0.7638      1.071     -0.713      0.476      -2.863       1.336
    q3:husd                -1.4076      0.919     -1.532      0.125      -3.209       0.393
    q4:q5                1.546e-15   3.89e-15      0.397      0.691   -6.08e-15    9.18e-15
    q4:q6                  -1.5424      1.657     -0.931      0.352      -4.791       1.707
    q4:agelt35             -0.4581      1.268     -0.361      0.718      -2.945       2.029
    q4:agegt54             -1.2224      1.158     -1.056      0.291      -3.492       1.047
    q4:durable              0.6783      1.262      0.537      0.591      -1.796       3.153
    q4:lusd                -1.0288      1.068     -0.963      0.335      -3.123       1.065
    q4:husd                -1.4925      0.937     -1.594      0.111      -3.328       0.343
    q5:q6                   1.3365      2.816      0.475      0.635      -4.184       6.858
    q5:agelt35              1.6827      2.001      0.841      0.400      -2.240       5.606
    q5:agegt54              2.0830      1.616      1.289      0.198      -1.086       5.252
    q5:durable             -0.0469      2.059     -0.023      0.982      -4.083       3.989
    q5:lusd                 1.6443      1.704      0.965      0.335      -1.696       4.985
    q5:husd                -1.0430      1.411     -0.739      0.460      -3.809       1.723
    q6:agelt35             -2.9488      1.470     -2.005      0.045      -5.831      -0.066
    q6:agegt54              1.3773      1.406      0.979      0.327      -1.380       4.134
    q6:durable              2.2884      1.534      1.492      0.136      -0.718       5.295
    q6:lusd                 1.7789      1.140      1.561      0.119      -0.456       4.013
    q6:husd                 0.4013      1.097      0.366      0.715      -1.750       2.552
    agelt35:agegt54        -0.4116      1.265     -0.325      0.745      -2.892       2.069
    agelt35:durable        -1.0104      1.380     -0.732      0.464      -3.716       1.695
    agelt35:lusd           -2.3812      1.181     -2.016      0.044      -4.697      -0.066
    agelt35:husd           -0.2792      1.097     -0.255      0.799      -2.429       1.870
    agegt54:durable              0          0        nan        nan           0           0
    agegt54:lusd            0.1624      1.093      0.149      0.882      -1.980       2.305
    agegt54:husd           -3.0090      0.894     -3.366      0.001      -4.762      -1.256
    durable:lusd           -2.3889      1.166     -2.049      0.040      -4.674      -0.103
    durable:husd           -0.0826      1.109     -0.074      0.941      -2.256       2.091
    lusd:husd                    0          0        nan        nan           0           0
    ==============================================================================
    Omnibus:                     1211.624   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              374.605
    Skew:                           0.397   Prob(JB):                     4.53e-82
    Kurtosis:                       2.039   Cond. No.                     1.46e+16
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 5.81e-29. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    


```python
#3 Interactive Regression Adjustment (IRA)
ira_model = sm.OLS.from_formula('inuidur1 ~ tg +tg*((female+black+othrace+C(dep)+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)**2)+(female+black+othrace+C(dep)+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)**2', data=Penn_filtered)
ira_results = ira_model.fit()
print(ira_results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               inuidur1   R-squared:                       0.076
    Model:                            OLS   Adj. R-squared:                  0.040
    Method:                 Least Squares   F-statistic:                     2.126
    Date:                Wed, 24 Apr 2024   Prob (F-statistic):           1.90e-18
    Time:                        00:15:25   Log-Likelihood:                -21592.
    No. Observations:                5782   AIC:                         4.362e+04
    Df Residuals:                    5566   BIC:                         4.505e+04
    Df Model:                         215                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                  9.8278      0.913     10.761      0.000       8.037      11.618
    C(dep)[T.1]                0.8863      1.654      0.536      0.592      -2.357       4.130
    C(dep)[T.2]               -0.4910      1.448     -0.339      0.735      -3.330       2.348
    tg                         0.2588      1.346      0.192      0.848      -2.380       2.897
    tg:C(dep)[T.1]             1.6405      2.384      0.688      0.491      -3.034       6.315
    tg:C(dep)[T.2]             0.7253      2.079      0.349      0.727      -3.350       4.801
    female                     0.8322      1.000      0.832      0.405      -1.128       2.793
    female:C(dep)[T.1]        -1.7359      1.263     -1.374      0.169      -4.212       0.740
    female:C(dep)[T.2]         2.6122      1.104      2.366      0.018       0.448       4.777
    black                      1.7183      1.893      0.908      0.364      -1.993       5.429
    black:C(dep)[T.1]         -0.7529      1.892     -0.398      0.691      -4.463       2.957
    black:C(dep)[T.2]         -1.8788      1.758     -1.068      0.285      -5.326       1.569
    othrace                   -8.4105      8.911     -0.944      0.345     -25.879       9.058
    othrace:C(dep)[T.1]      -14.8831     18.688     -0.796      0.426     -51.519      21.753
    othrace:C(dep)[T.2]      -12.2262      7.936     -1.541      0.123     -27.785       3.332
    q2                         1.6599      1.242      1.336      0.182      -0.776       4.096
    C(dep)[T.1]:q2            -0.0935      1.795     -0.052      0.958      -3.612       3.425
    C(dep)[T.2]:q2             2.7504      1.473      1.867      0.062      -0.138       5.639
    q3                         0.3596      1.164      0.309      0.757      -1.922       2.642
    C(dep)[T.1]:q3            -0.1322      1.642     -0.081      0.936      -3.352       3.087
    C(dep)[T.2]:q3             1.8500      1.465      1.263      0.207      -1.022       4.722
    q4                         0.7980      1.190      0.670      0.503      -1.536       3.131
    C(dep)[T.1]:q4            -0.5571      1.634     -0.341      0.733      -3.760       2.646
    C(dep)[T.2]:q4             2.3619      1.507      1.568      0.117      -0.591       5.315
    q5                         0.6709      1.688      0.397      0.691      -2.638       3.980
    C(dep)[T.1]:q5             3.4673      2.596      1.336      0.182      -1.622       8.556
    C(dep)[T.2]:q5             4.0192      2.073      1.938      0.053      -0.046       8.084
    q6                         1.8037      2.315      0.779      0.436      -2.736       6.343
    C(dep)[T.1]:q6            -1.0120      1.909     -0.530      0.596      -4.754       2.730
    C(dep)[T.2]:q6             0.0813      1.675      0.049      0.961      -3.202       3.365
    agelt35                    4.3101      1.654      2.606      0.009       1.068       7.552
    C(dep)[T.1]:agelt35        0.1555      1.716      0.091      0.928      -3.208       3.519
    C(dep)[T.2]:agelt35        2.1289      4.245      0.502      0.616      -6.192      10.450
    agegt54                    2.8409      1.322      2.148      0.032       0.249       5.433
    C(dep)[T.1]:agegt54        2.9014      1.659      1.749      0.080      -0.351       6.153
    C(dep)[T.2]:agegt54        0.2003      1.409      0.142      0.887      -2.562       2.962
    durable                    1.0520      1.672      0.629      0.529      -2.225       4.329
    C(dep)[T.1]:durable       -0.4351      1.980     -0.220      0.826      -4.316       3.446
    C(dep)[T.2]:durable       -1.7438      1.632     -1.068      0.285      -4.944       1.457
    lusd                       2.2455      1.196      1.878      0.060      -0.099       4.590
    C(dep)[T.1]:lusd           1.4976      1.571      0.954      0.340      -1.581       4.577
    C(dep)[T.2]:lusd           0.2628      1.398      0.188      0.851      -2.477       3.003
    husd                       3.6833      1.053      3.497      0.000       1.619       5.748
    C(dep)[T.1]:husd           2.2996      1.395      1.648      0.099      -0.436       5.035
    C(dep)[T.2]:husd          -1.7046      1.209     -1.410      0.159      -4.074       0.665
    female:black              -3.0376      1.210     -2.510      0.012      -5.410      -0.665
    female:othrace            -5.4533      8.357     -0.653      0.514     -21.837      10.930
    female:q2                 -0.4863      1.120     -0.434      0.664      -2.682       1.710
    female:q3                  2.6126      1.071      2.439      0.015       0.513       4.713
    female:q4                  1.4543      1.076      1.352      0.176      -0.654       3.563
    female:q5                  2.5397      1.672      1.519      0.129      -0.738       5.818
    female:q6                 -1.8016      1.320     -1.365      0.172      -4.390       0.787
    female:agelt35             2.5594      1.322      1.936      0.053      -0.032       5.151
    female:agegt54             0.2609      1.196      0.218      0.827      -2.085       2.606
    female:durable            -2.0163      1.226     -1.645      0.100      -4.419       0.386
    female:lusd               -0.2216      1.059     -0.209      0.834      -2.297       1.854
    female:husd                0.1037      0.893      0.116      0.908      -1.647       1.854
    black:othrace          -3.534e-13   2.22e-13     -1.593      0.111   -7.88e-13    8.16e-14
    black:q2                  -1.0227      1.817     -0.563      0.574      -4.585       2.540
    black:q3                  -1.6886      1.605     -1.052      0.293      -4.835       1.458
    black:q4                  -0.1706      1.643     -0.104      0.917      -3.391       3.050
    black:q5                  -4.5314      2.411     -1.880      0.060      -9.258       0.195
    black:q6                   0.5765      2.744      0.210      0.834      -4.804       5.957
    black:agelt35              2.2396      2.250      0.995      0.320      -2.171       6.650
    black:agegt54              4.3302      1.900      2.279      0.023       0.605       8.055
    black:durable             -2.9957      2.080     -1.440      0.150      -7.073       1.082
    black:lusd                 6.5392      6.299      1.038      0.299      -5.810      18.888
    black:husd                -1.8614      1.523     -1.222      0.222      -4.848       1.125
    othrace:q2                22.0131     12.306      1.789      0.074      -2.111      46.137
    othrace:q3                 3.9034     11.132      0.351      0.726     -17.919      25.726
    othrace:q4                14.6777      8.317      1.765      0.078      -1.626      30.981
    othrace:q5                 0.6847      7.944      0.086      0.931     -14.889      16.258
    othrace:q6              8.766e-14   4.57e-14      1.918      0.055   -1.94e-15    1.77e-13
    othrace:agelt35          -16.7377      9.922     -1.687      0.092     -36.188       2.712
    othrace:agegt54           15.7364      7.346      2.142      0.032       1.336      30.137
    othrace:durable           19.3054     10.332      1.868      0.062      -0.950      39.561
    othrace:lusd              -8.2809     15.229     -0.544      0.587     -38.136      21.574
    othrace:husd               3.7011      7.467      0.496      0.620     -10.937      18.339
    q2:q3                  -6.711e-14   5.45e-14     -1.232      0.218   -1.74e-13    3.97e-14
    q2:q4                   8.587e-15   2.81e-14      0.306      0.760   -4.65e-14    6.37e-14
    q2:q5                   4.853e-14   6.13e-14      0.792      0.429   -7.16e-14    1.69e-13
    q2:q6                      0.4703      2.168      0.217      0.828      -3.780       4.721
    q2:agelt35                 0.6303      1.838      0.343      0.732      -2.974       4.234
    q2:agegt54                -4.2332      1.520     -2.785      0.005      -7.213      -1.254
    q2:durable                -0.6013      1.783     -0.337      0.736      -4.097       2.894
    q2:lusd                   -1.8042      1.491     -1.210      0.226      -4.726       1.118
    q2:husd                   -0.4709      1.289     -0.365      0.715      -2.999       2.057
    q3:q4                  -3.704e-14   3.54e-14     -1.047      0.295   -1.06e-13    3.23e-14
    q3:q5                  -3.804e-14   5.45e-14     -0.698      0.485   -1.45e-13    6.88e-14
    q3:q6                     -1.7519      2.201     -0.796      0.426      -6.066       2.562
    q3:agelt35                 4.3564      1.664      2.617      0.009       1.093       7.619
    q3:agegt54                -1.2102      1.474     -0.821      0.412      -4.099       1.679
    q3:durable                 0.7455      1.723      0.433      0.665      -2.632       4.123
    q3:lusd                   -1.0258      1.434     -0.716      0.474      -3.836       1.785
    q3:husd                   -2.3277      1.230     -1.893      0.058      -4.739       0.083
    q4:q5                  -3.224e-15    5.2e-14     -0.062      0.951   -1.05e-13    9.86e-14
    q4:q6                     -3.1074      2.232     -1.392      0.164      -7.482       1.267
    q4:agelt35                -0.1259      1.624     -0.078      0.938      -3.309       3.057
    q4:agegt54                -2.1391      1.513     -1.413      0.158      -5.106       0.828
    q4:durable                -0.0862      1.681     -0.051      0.959      -3.382       3.209
    q4:lusd                   -1.3741      1.427     -0.963      0.336      -4.172       1.423
    q4:husd                   -1.3654      1.263     -1.081      0.280      -3.842       1.111
    q5:q6                     -0.5410      3.714     -0.146      0.884      -7.822       6.740
    q5:agelt35                 5.9967      2.525      2.375      0.018       1.048      10.946
    q5:agegt54                 0.5761      2.143      0.269      0.788      -3.625       4.777
    q5:durable                -0.3738      2.744     -0.136      0.892      -5.753       5.005
    q5:lusd                   -0.5365      2.194     -0.245      0.807      -4.838       3.765
    q5:husd                   -1.9544      1.863     -1.049      0.294      -5.607       1.698
    q6:agelt35                -3.8715      1.920     -2.016      0.044      -7.636      -0.107
    q6:agegt54                 0.6546      1.889      0.347      0.729      -3.048       4.357
    q6:durable                 3.0858      2.086      1.479      0.139      -1.004       7.176
    q6:lusd                    1.9083      1.495      1.277      0.202      -1.022       4.839
    q6:husd                    0.2396      1.444      0.166      0.868      -2.592       3.071
    agelt35:agegt54           -1.5831      1.702     -0.930      0.352      -4.919       1.753
    agelt35:durable           -2.8127      1.880     -1.496      0.135      -6.497       0.872
    agelt35:lusd              -4.7310      1.553     -3.046      0.002      -7.776      -1.686
    agelt35:husd              -3.5456      1.492     -2.376      0.018      -6.471      -0.621
    agegt54:durable         1.621e-15   3.03e-14      0.054      0.957   -5.78e-14     6.1e-14
    agegt54:lusd               0.2336      1.456      0.160      0.873      -2.620       3.087
    agegt54:husd              -2.5008      1.178     -2.122      0.034      -4.811      -0.191
    durable:lusd              -3.1070      1.586     -1.960      0.050      -6.215       0.001
    durable:husd               1.1513      1.484      0.776      0.438      -1.758       4.061
    lusd:husd              -1.748e-14   2.47e-14     -0.709      0.479   -6.59e-14    3.09e-14
    tg:female                  1.0629      1.500      0.709      0.478      -1.877       4.003
    tg:female:C(dep)[T.1]      1.3182      1.938      0.680      0.497      -2.482       5.118
    tg:female:C(dep)[T.2]     -4.1448      1.713     -2.420      0.016      -7.502      -0.787
    tg:black                  -4.3026      2.876     -1.496      0.135      -9.940       1.335
    tg:black:C(dep)[T.1]       3.0436      2.959      1.029      0.304      -2.758       8.845
    tg:black:C(dep)[T.2]       3.4423      2.748      1.252      0.210      -1.946       8.830
    tg:othrace                20.2365     12.663      1.598      0.110      -4.588      45.061
    tg:othrace:C(dep)[T.1]    26.2446     23.229      1.130      0.259     -19.294      71.783
    tg:othrace:C(dep)[T.2]    15.6799     18.447      0.850      0.395     -20.484      51.844
    tg:q2                      1.4504      1.879      0.772      0.440      -2.234       5.135
    tg:C(dep)[T.1]:q2         -0.7763      2.635     -0.295      0.768      -5.942       4.389
    tg:C(dep)[T.2]:q2         -0.5013      2.300     -0.218      0.827      -5.011       4.008
    tg:q3                     -0.9658      1.746     -0.553      0.580      -4.390       2.458
    tg:C(dep)[T.1]:q3         -2.3846      2.489     -0.958      0.338      -7.265       2.495
    tg:C(dep)[T.2]:q3         -0.6663      2.234     -0.298      0.766      -5.046       3.713
    tg:q4                     -0.2957      1.770     -0.167      0.867      -3.766       3.175
    tg:C(dep)[T.1]:q4         -0.3579      2.497     -0.143      0.886      -5.253       4.537
    tg:C(dep)[T.2]:q4         -0.3719      2.321     -0.160      0.873      -4.922       4.178
    tg:q5                      1.6798      2.673      0.628      0.530      -3.561       6.921
    tg:C(dep)[T.1]:q5         -8.2718      4.211     -1.964      0.050     -16.526      -0.017
    tg:C(dep)[T.2]:q5         -1.5401      3.338     -0.461      0.645      -8.084       5.004
    tg:q6                     -3.7391      3.474     -1.076      0.282     -10.549       3.071
    tg:C(dep)[T.1]:q6          1.0637      2.999      0.355      0.723      -4.816       6.943
    tg:C(dep)[T.2]:q6         -1.4938      2.604     -0.574      0.566      -6.599       3.611
    tg:agelt35                 1.2349      2.572      0.480      0.631      -3.807       6.277
    tg:C(dep)[T.1]:agelt35    -3.0556      2.690     -1.136      0.256      -8.330       2.219
    tg:C(dep)[T.2]:agelt35     0.3993      7.602      0.053      0.958     -14.503      15.301
    tg:agegt54                -3.1443      1.994     -1.577      0.115      -7.053       0.764
    tg:C(dep)[T.1]:agegt54    -1.9927      2.469     -0.807      0.420      -6.834       2.848
    tg:C(dep)[T.2]:agegt54     0.5346      2.309      0.231      0.817      -3.993       5.062
    tg:durable                -2.3492      2.605     -0.902      0.367      -7.456       2.758
    tg:C(dep)[T.1]:durable    -3.1685      3.068     -1.033      0.302      -9.184       2.847
    tg:C(dep)[T.2]:durable     0.9439      2.542      0.371      0.710      -4.039       5.926
    tg:lusd                   -1.1650      1.775     -0.656      0.512      -4.645       2.315
    tg:C(dep)[T.1]:lusd        0.1659      2.374      0.070      0.944      -4.489       4.821
    tg:C(dep)[T.2]:lusd        1.1926      2.151      0.554      0.579      -3.025       5.410
    tg:husd                   -0.4942      1.569     -0.315      0.753      -3.571       2.582
    tg:C(dep)[T.1]:husd       -1.8577      2.100     -0.885      0.376      -5.974       2.259
    tg:C(dep)[T.2]:husd        1.7230      1.843      0.935      0.350      -1.890       5.336
    tg:female:black           -0.9525      1.941     -0.491      0.624      -4.758       2.853
    tg:female:othrace         -0.9730     12.315     -0.079      0.937     -25.115      23.169
    tg:female:q2               0.1677      1.766      0.095      0.924      -3.294       3.629
    tg:female:q3               0.5761      1.641      0.351      0.726      -2.641       3.793
    tg:female:q4              -1.0119      1.658     -0.610      0.542      -4.262       2.238
    tg:female:q5              -4.7543      2.690     -1.767      0.077     -10.028       0.519
    tg:female:q6               1.6985      2.167      0.784      0.433      -2.549       5.946
    tg:female:agelt35         -4.6953      2.069     -2.270      0.023      -8.751      -0.640
    tg:female:agegt54          1.8065      1.897      0.952      0.341      -1.912       5.525
    tg:female:durable          1.2556      1.857      0.676      0.499      -2.385       4.896
    tg:female:lusd            -1.9117      1.609     -1.188      0.235      -5.066       1.242
    tg:female:husd            -0.8453      1.374     -0.615      0.538      -3.539       1.849
    tg:black:othrace       -1.392e-14   1.69e-14     -0.824      0.410    -4.7e-14    1.92e-14
    tg:black:q2               -0.9346      2.739     -0.341      0.733      -6.305       4.436
    tg:black:q3                1.7119      2.502      0.684      0.494      -3.193       6.617
    tg:black:q4                0.0516      2.684      0.019      0.985      -5.210       5.313
    tg:black:q5                2.6934      3.972      0.678      0.498      -5.093      10.480
    tg:black:q6               -1.6665      4.435     -0.376      0.707     -10.361       7.028
    tg:black:agelt35          -4.0798      3.886     -1.050      0.294     -11.699       3.539
    tg:black:agegt54          -6.1497      2.978     -2.065      0.039     -11.988      -0.311
    tg:black:durable           4.1645      3.296      1.263      0.206      -2.297      10.627
    tg:black:lusd             -5.1784      8.466     -0.612      0.541     -21.774      11.418
    tg:black:husd              3.5732      2.289      1.561      0.119      -0.914       8.060
    tg:othrace:q2           3.364e-15   2.33e-15      1.442      0.149   -1.21e-15    7.94e-15
    tg:othrace:q3            -16.0881     12.737     -1.263      0.207     -41.058       8.882
    tg:othrace:q4            -24.9350     12.887     -1.935      0.053     -50.199       0.329
    tg:othrace:q5             -6.9000     17.018     -0.405      0.685     -40.263      26.463
    tg:othrace:q6           8.911e-16   2.88e-15      0.310      0.757   -4.75e-15    6.53e-15
    tg:othrace:agelt35        -8.4568     14.274     -0.592      0.554     -36.440      19.527
    tg:othrace:agegt54       -10.7719     11.994     -0.898      0.369     -34.285      12.742
    tg:othrace:durable       -23.0281     11.636     -1.979      0.048     -45.839      -0.218
    tg:othrace:lusd          1.22e-15   2.39e-15      0.511      0.609   -3.46e-15     5.9e-15
    tg:othrace:husd          -10.4425     11.744     -0.889      0.374     -33.466      12.581
    tg:q2:q3               -6.787e-16    1.7e-15     -0.400      0.689   -4.01e-15    2.65e-15
    tg:q2:q4                 1.93e-15   2.06e-15      0.936      0.349   -2.11e-15    5.97e-15
    tg:q2:q5                2.827e-15   1.78e-15      1.591      0.112   -6.57e-16    6.31e-15
    tg:q2:q6                   2.1337      3.278      0.651      0.515      -4.292       8.559
    tg:q2:agelt35             -4.7522      2.880     -1.650      0.099     -10.398       0.894
    tg:q2:agegt54              2.4056      2.369      1.015      0.310      -2.240       7.051
    tg:q2:durable              1.0886      2.753      0.395      0.693      -4.309       6.486
    tg:q2:lusd                 0.6183      2.267      0.273      0.785      -3.826       5.063
    tg:q2:husd                -2.5597      1.988     -1.287      0.198      -6.458       1.338
    tg:q3:q4               -6.752e-16   9.14e-16     -0.739      0.460   -2.47e-15    1.12e-15
    tg:q3:q5                2.325e-15   1.73e-15      1.340      0.180   -1.08e-15    5.73e-15
    tg:q3:q6                   4.0395      3.443      1.173      0.241      -2.709      10.788
    tg:q3:agelt35             -5.0812      2.615     -1.943      0.052     -10.208       0.045
    tg:q3:agegt54              2.7794      2.250      1.236      0.217      -1.631       7.189
    tg:q3:durable              1.2401      2.606      0.476      0.634      -3.870       6.350
    tg:q3:lusd                 0.1689      2.182      0.077      0.938      -4.109       4.447
    tg:q3:husd                 1.6956      1.867      0.908      0.364      -1.965       5.356
    tg:q4:q5                3.179e-16   5.66e-16      0.562      0.574   -7.91e-16    1.43e-15
    tg:q4:q6                   3.6858      3.441      1.071      0.284      -3.061      10.432
    tg:q4:agelt35             -1.1383      2.654     -0.429      0.668      -6.342       4.065
    tg:q4:agegt54              1.9636      2.384      0.824      0.410      -2.711       6.638
    tg:q4:durable              1.7981      2.594      0.693      0.488      -3.287       6.884
    tg:q4:lusd                 0.5818      2.170      0.268      0.789      -3.673       4.836
    tg:q4:husd                -0.7864      1.903     -0.413      0.679      -4.516       2.943
    tg:q5:q6                   3.1953      5.950      0.537      0.591      -8.468      14.859
    tg:q5:agelt35             -9.3825      4.299     -2.183      0.029     -17.810      -0.955
    tg:q5:agegt54              3.8070      3.362      1.132      0.258      -2.785      10.399
    tg:q5:durable             -0.3152      4.269     -0.074      0.941      -8.684       8.053
    tg:q5:lusd                 4.6450      3.563      1.304      0.192      -2.340      11.631
    tg:q5:husd                 0.8381      2.903      0.289      0.773      -4.853       6.529
    tg:q6:agelt35              1.8250      3.051      0.598      0.550      -4.155       7.805
    tg:q6:agegt54              1.3575      2.880      0.471      0.637      -4.288       7.003
    tg:q6:durable             -2.1416      3.192     -0.671      0.502      -8.400       4.117
    tg:q6:lusd                -0.1260      2.357     -0.053      0.957      -4.747       4.495
    tg:q6:husd                 1.4838      2.272      0.653      0.514      -2.971       5.938
    tg:agelt35:agegt54         1.5353      2.602      0.590      0.555      -3.566       6.636
    tg:agelt35:durable         4.2777      2.860      1.495      0.135      -1.330       9.885
    tg:agelt35:lusd            5.9495      2.446      2.432      0.015       1.155      10.744
    tg:agelt35:husd            6.7085      2.270      2.956      0.003       2.259      11.158
    tg:agegt54:durable              0          0        nan        nan           0           0
    tg:agegt54:lusd           -0.4620      2.233     -0.207      0.836      -4.839       3.915
    tg:agegt54:husd           -1.6667      1.836     -0.908      0.364      -5.265       1.932
    tg:durable:lusd            1.4912      2.396      0.622      0.534      -3.207       6.189
    tg:durable:husd           -3.2968      2.292     -1.438      0.150      -7.791       1.197
    tg:lusd:husd                    0          0        nan        nan           0           0
    ==============================================================================
    Omnibus:                     1021.804   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              354.529
    Skew:                           0.394   Prob(JB):                     1.04e-77
    Kurtosis:                       2.077   Cond. No.                     2.27e+16
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.67e-29. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    


```python
#4 Interactive Regression Adjustment Using Lasso (IRA using Lasso)

from hdm import PyRlasso


X = pd.get_dummies(Penn_filtered[['tg', 'female', 'black', 'agelt35', 'dep']], drop_first=True, columns=['dep'])
X['intercept'] = 1
y = Penn_filtered['inuidur1']

lasso_model = PyRlasso(X.values, y.values, post=True)
lasso_results = lasso_model.fit()
print(f"Lasso Coefficients: {lasso_results.coef_}")
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[13], line 3
          1 #4 Interactive Regression Adjustment Using Lasso (IRA using Lasso)
    ----> 3 from hdm import PyRlasso
          6 X = pd.get_dummies(Penn_filtered[['tg', 'female', 'black', 'agelt35', 'dep']], drop_first=True, columns=['dep'])
          7 X['intercept'] = 1
    

    ModuleNotFoundError: No module named 'hdm'


# Plot coefficients


```python
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np

coefs = {'tgdep': ira_model.params['tg:C(dep)[T.1]'],
         'tgfem': ira_model.params['tg:female'],
         'tgblack': ira_model.params['tg:black'],
        'tgagelt35': ira_model.params['tg:agelt35']}

ses = {'tgdep': ira_model.bse['tg:C(dep)[T.1]'],
         'tgfem': ira_model.bse['tg:female'],
         'tgblack': ira_model.bse['tg:black'],
        'tgagelt35': ira_model.bse['tg:agelt35']}

plt.errorbar(coefs.keys(), coefs.values(), yerr=1.96 * np.array(list(ses.values())), fmt='o', capsize=5)
plt.axhline(y=0, color='gray', linestyle='--')
plt.ylabel('Coefficient')
plt.title('Coefficientes IRA')
plt.show()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[14], line 5
          2 from matplotlib import pyplot as plt
          3 import numpy as np
    ----> 5 coefs = {'tgdep': ira_model.params['tg:C(dep)[T.1]'],
          6          'tgfem': ira_model.params['tg:female'],
          7          'tgblack': ira_model.params['tg:black'],
          8         'tgagelt35': ira_model.params['tg:agelt35']}
         10 ses = {'tgdep': ira_model.bse['tg:C(dep)[T.1]'],
         11          'tgfem': ira_model.bse['tg:female'],
         12          'tgblack': ira_model.bse['tg:black'],
         13         'tgagelt35': ira_model.bse['tg:agelt35']}
         15 plt.errorbar(coefs.keys(), coefs.values(), yerr=1.96 * np.array(list(ses.values())), fmt='o', capsize=5)
    

    AttributeError: 'OLS' object has no attribute 'params'



```python

```

## A crash course in good and bad controls

In this section, we will explore different scenarios where we need to decide whether the inclusion of a control variable, denoted by _Z_, will help (or not) to improve the estimation of the **average treatment effect** (ATE) of treatment _X_ on outcome _Y_. The effect of observed variables will be represented by a continuous line, while that of unobserved variables will be represented by and discontinuous line.


```python
# Libraries
import pandas as pd, numpy as np, statsmodels.formula.api as smf
from causalgraphicalmodels import CausalGraphicalModel
from statsmodels.iolib.summary2 import summary_col
```

#### Good control (Blocking back-door paths)

**Model 1** 

We will assume that _X_ measures whether or not the student attends the extra tutoring session, that affects the student's grade (_Y_). Then, we have another observable variable, as hours of the student sleep (_Z_), that impacts _X_ and _Y_. Theory says that when controlling by _Z_, we block the back-door path from _X_ to _Y_. Thus, we see that in the second regression, the coefficient of _X_ is closer to the real one (2.9898 ≈ 3).


```python
sprinkler = CausalGraphicalModel(nodes=["Z","Y","X"],
                                 edges=[("X","Y"),
                                        ("Z","X"),
                                        ("Z","Y")])
sprinkler.draw()
```




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p1_output_28_0.svg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    




```python
np.random.seed(24) # set seed

# Generate data
n = 1000 # sample size
Z = np.random.normal(0,1, 1000).reshape((1000, 1))
X = 5 * Z + np.random.normal(0, 1, 1000).reshape((1000, 1))
Y = 3 * X + 1.5 * Z + np.random.normal(0, 1, 1000).reshape((1000, 1))

# Create dataframe
D = np.hstack((Z, X, Y))
data = pd.DataFrame(D, columns = ["Z", "X", "Y"])
```


```python
# Regressions
no_control = smf.ols("Y ~ X", data=data).fit()        # Wrong, not controlling by the confounder Z
using_control = smf.ols("Y ~ X + Z", data=data).fit() # Correct

# Summary results
dfoutput = summary_col([no_control, using_control], stars=True)
print(dfoutput)
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    
    ==================================
                      Y I       Y II  
    ----------------------------------
    Intercept      -0.0121   -0.0283  
                   (0.0322)  (0.0306) 
    R-squared      0.9964    0.9967   
    R-squared Adj. 0.9964    0.9967   
    X              3.2928*** 2.9643***
                   (0.0063)  (0.0315) 
    Z                        1.6891***
                             (0.1590) 
    ==================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    

**Model 2** 

We will assume that _X_ stands for the police salaries that affect the crime rate (_Y_). Then, we have another observable variable, as the policemen's supply (_Z_), that impacts _X_ but not _Y_. And, additionally, we know that there is an unobservable variable (denoted by a _•_), as the preference for maintaining civil order, that affects _Z_ and _Y_. The theory says that when controlling by _Z_, we block (some) of the unobservable variable’s back-door path from _X_ to _Y_. Thus, we see that in the second regression, the coefficient of _X_ is equal to the real one (0.5).


```python
sprinkler = CausalGraphicalModel(nodes=["Z","Y","X"],
                                 edges=[("Z","X"),
                                        ("X","Y")],
                                 latent_edges=[("Z","Y")])
sprinkler.draw()
```




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p2_output_32_0.svg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    




```python
np.random.seed(24) # set seed

n = 1000
U = np.random.normal(0, 1, 1000).reshape((1000, 1))
Z = 7 * U + np.random.normal(0, 1, 1000).reshape((1000, 1))
X = 2 * Z + np.random.normal(0, 1, 1000).reshape((1000, 1))
Y = 0.5 * X + 0.2 * U + np.random.normal(0, 1, 1000).reshape((1000, 1))

# Create dataframe
D = np.hstack((U, Z, X, Y))
data = pd.DataFrame(D, columns = ["U", "Z", "X", "Y"])
```


```python
# Regressions
no_control = smf.ols("Y ~ X", data=data).fit()
using_control = smf.ols("Y ~ X + Z", data=data).fit()

# Summary results
dfoutput = summary_col([no_control,using_control], stars=True)
print(dfoutput)
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    
    ==================================
                      Y I       Y II  
    ----------------------------------
    Intercept      -0.0003   -0.0006  
                   (0.0312)  (0.0312) 
    R-squared      0.9820    0.9820   
    R-squared Adj. 0.9820    0.9820   
    X              0.5104*** 0.5000***
                   (0.0022)  (0.0323) 
    Z                        0.0209   
                             (0.0649) 
    ==================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    

#### Bad Control (M-bias)

**Model 7** 

Let us suppose that _X_ stands for a job training program aimed at reducing unemployment. Then, there is a first unobserved confounder, which could be the planning effort and good design of the job program (right _•_) that impacts directly on the participation in job training programs (_X_) and the proximity of job programs (that would be the bad control _Z_). Furthermore, we have another unobserved confounder (left _•_), as the soft skills of the unemployed, that affects the employment status of individuals (_Y_) and the likelihood of beeing in a job training program that is closer (_Z_). That is why including _Z_ in the second regression makes _X_ coefficient value further to the real one.


```python
sprinkler = CausalGraphicalModel(nodes=["Z","Y","X"],
                                 edges=[("X","Y")],
                                 latent_edges=[("X","Z"),("Z","Y")]
                                )
sprinkler.draw()
```




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p2_output_37_0.svg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    




```python
np.random.seed(24) # set seed

n = 1000
U_1 = np.random.normal(0, 1, 1000).reshape((1000, 1))
U_2 = np.random.normal(0, 1, 1000).reshape((1000, 1))

Z = 0.3 * U_1 + 0.9 * U_2 + np.random.normal(0, 1, 1000).reshape((1000, 1)) # generate Z
X = 4 * U_1 + np.random.normal(0, 1, 1000).reshape((1000, 1))
Y = 3 * X + U_2 + np.random.normal(0, 1, 1000).reshape((1000, 1))

# Create dataframe
D = np.hstack((U_1, U_2, Z, X, Y))
data = pd.DataFrame(D, columns = ["U_1", "U_2", "Z", "X", "Y"])
```


```python
# Regressions
no_control = smf.ols("Y ~ X", data=data).fit()
using_control = smf.ols("Y ~ X + Z", data=data).fit()

# Summary results
dfoutput = summary_col([no_control, using_control], stars=True)
print(dfoutput)
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    
    ==================================
                      Y I       Y II  
    ----------------------------------
    Intercept      -0.0549   -0.0251  
                   (0.0422)  (0.0384) 
    R-squared      0.9884    0.9904   
    R-squared Adj. 0.9884    0.9904   
    X              2.9879*** 2.9596***
                   (0.0102)  (0.0095) 
    Z                        0.4349***
                             (0.0300) 
    ==================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    

#### Neutral Control (possibly good for precision)

**Model 8** 

In this scenario, we will assume that _X_ represents the implementation of a new government policy to provide subsidies and guidance for small companies. There is another variable, _Z_, that stands for the % inflation rate. And both _X_ and _Z_ affect _Y_, which represents the GDP growth rate of the country. Then, even if _Z_ does not impact _X_, its inclusion improves the precision of the ATE estimator (8.5643 is closer to 8.6).


```python
sprinkler = CausalGraphicalModel(nodes=["Z","Y","X"],
                                 edges=[("Z","Y"),("X","Y")])
sprinkler.draw()
```




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p2_output_42_0.svg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    




```python
np.random.seed(24) # set seed

n = 1000

Z = np.random.normal(0, 1, 1000).reshape((1000, 1))
X = np.random.normal(0, 1, 1000).reshape((1000, 1))
Y = 8.6 * X + 5 * Z + np.random.normal(0, 1, 1000).reshape((1000, 1))

# Create dataframe
D = np.hstack((Z, X, Y))
data = pd.DataFrame(D, columns = ["Z", "X", "Y"])
```


```python
# Regressions
no_control = smf.ols("Y ~ X", data=data).fit()
using_control = smf.ols("Y ~ X + Z", data=data).fit()

# Summary results
dfoutput = summary_col([no_control, using_control], stars=True)
print(dfoutput)
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    
    ==================================
                      Y I       Y II  
    ----------------------------------
    Intercept      0.0289    -0.0283  
                   (0.1636)  (0.0306) 
    R-squared      0.7109    0.9899   
    R-squared Adj. 0.7107    0.9899   
    X              8.3355*** 8.5643***
                   (0.1682)  (0.0315) 
    Z                        5.0108***
                             (0.0302) 
    ==================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    

#### Bad Controls (Bias amplification)

**Model 10** 

Let us assume that _X_ measures the implementation of a housing program for young adults buying their first house, which impacts the average housing prices (_Y_). There is another observable variable, _Z_, that measures the expenditure of the program and affects only _X_. Also, there is an unobservable variable (represented by a •) that represents the preference of young adults to move from their parent's house and impacts only _X_ and _Y_. Therefore, the inclusion of _Z_ will "amplify the bias" of (•) on _X_, so the ATE estimator will be worse. We can see that in the second regression, the estimator (0.8241) is much farther from the real value (0.8).


```python
sprinkler = CausalGraphicalModel(nodes=["Z", "Y", "X"],
                                 edges=[("Z", "X"), ("X", "Y")],
                                 latent_edges=[("X", "Y")])
sprinkler.draw()
```




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p2_output_47_0.svg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    




```python
np.random.seed(24) # set seed

n = 1000
U = np.random.normal(0, 1, 1000).reshape((1000, 1))
Z = np.random.normal(0, 1, 1000).reshape((1000, 1))
X = 3 * Z + 6 * U + np.random.normal(0, 1, 1000).reshape((1000, 1))
Y = 0.8 * X + 0.2 * U + np.random.normal(0, 1, 1000).reshape((1000, 1))

# Create dataframe
D = np.hstack((U, Z, X, Y))
data = pd.DataFrame(D, columns = ["U", "Z", "X", "Y"])
```


```python
# Regressions
no_control = smf.ols("Y ~ X" , data=data).fit()
using_control = smf.ols("Y ~ X + Z" , data=data).fit()

# Summary results
print(summary_col([no_control, using_control], stars=True))
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    
    ==================================
                      Y I       Y II  
    ----------------------------------
    Intercept      0.0021    -0.0013  
                   (0.0313)  (0.0312) 
    R-squared      0.9686    0.9687   
    R-squared Adj. 0.9685    0.9687   
    X              0.8195*** 0.8241***
                   (0.0047)  (0.0051) 
    Z                        -0.0812**
                             (0.0349) 
    ==================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    

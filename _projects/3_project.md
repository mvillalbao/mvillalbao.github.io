---
layout: page
title: Lab 3
description: Third lab assignment for the Causal AI course
img: assets/img/p3_output_17_4.png
importance: 3
category: Causal AI course
giscus_comments: false
---

# Workgroup 3

**Group 3**: Valerie Dube, Erzo Garay, Juan Marcos Guerrero y Matías Villalba,




## 2. Code Section

### 2.1. Orthogonal Learning


```python
import hdmpy
import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import colors
from multiprocess import Pool
import seaborn as sns
import time
```

### Simulation Design

We are going to simulate 3 different trials to show the properties we talked about orthogonal learning.

For that we first define a function that runs a single observation of our simulation.


```python
def simulate_once(seed):
    import numpy as np  # Ensure numpy is imported within the function
    import hdmpy
    import statsmodels.api as sm
    np.random.seed(seed)
    n = 100
    p = 100
    beta = ( 1 / (np.arange( 1, p + 1 ) ** 2 ) ).reshape( p , 1 )
    gamma = ( 1 / (np.arange( 1, p + 1 ) ** 2 ) ).reshape( p , 1 )

    mean = 0
    sd = 1
    X = np.random.normal( mean , sd, n * p ).reshape( n, p )

    D = ( X @ gamma ) + np.random.normal( mean , sd, n ).reshape( n, 1 )/4 # We reshape because in r when we sum a vecto with a matrix it sum by column
    Y = 10 * D + ( X @ beta ) + np.random.normal( mean , sd, n ).reshape( n, 1 )

    # single selection method
    r_lasso_estimation = hdmpy.rlasso( np.concatenate( ( D , X ) , axis  =  1 ) , Y , post = True ) # Regress main equation by lasso
    coef_array = r_lasso_estimation.est[ 'coefficients' ].iloc[ 2:, :].to_numpy()    # Get "X" coefficients 
    SX_IDs = np.where( coef_array != 0 )[0]

    # In case all X coefficients are zero, then regress Y on D
    if sum(SX_IDs) == 0 : 
        naive_coef = sm.OLS( Y , sm.add_constant(D) ).fit().summary2().tables[1].round(3).iloc[ 1, 0 ] 

    # Otherwise, then regress Y on X and D (but only in the selected coefficients)
    elif sum( SX_IDs ) > 0 :
        X_D = np.concatenate( ( D, X[:, SX_IDs ] ) , axis = 1 )
        naive_coef = sm.OLS( Y , sm.add_constant( X_D ) ).fit().summary2().tables[1].round(3).iloc[ 1, 0]

    # In both cases we save D coefficient
        
    # Regress residuals. 
    resY = hdmpy.rlasso( X , Y , post = False ).est[ 'residuals' ]
    resD = hdmpy.rlasso( X , D , post = False ).est[ 'residuals' ]
    orthogonal_coef = sm.OLS( resY , sm.add_constant( resD ) ).fit().summary2().tables[1].round(3).iloc[ 1, 0]

    return naive_coef, orthogonal_coef
```

Then we define a function that runs the simulation on its enterity, using parallel computing and the function we previously defined.


```python
def run_simulation(B):
    with Pool() as pool:
        results = pool.map(simulate_once, range(B))
    Naive = np.array([result[0] for result in results])
    Orthogonal = np.array([result[1] for result in results])
    return Naive, Orthogonal
```


```python
Orto_breaks = np.arange(8,12,0.2)
Naive_breaks = np.arange(8,12,0.2)
```

Next we run the simulations with 100, 1000, and 10000 iterations and plot the histograms for the Naive and Orthogonal estimations


```python
Bs = [100, 1000, 10000]

for B in Bs:
    start_time = time.time()
    Naive, Orthogonal = run_simulation(B)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    fig, axs = plt.subplots(1, 2, sharex=True, tight_layout=True)
    axs[0].hist(Orthogonal, range=(8,12), density=True, bins=Orto_breaks, color='lightblue')
    axs[1].hist(Naive, range=(8,12), density=True, bins=Naive_breaks, color='lightblue')
    sns.kdeplot(Orthogonal, ax=axs[0], color='darkblue')
    sns.kdeplot(Naive, ax=axs[1], color='darkblue')
    
    axs[0].axvline(x=np.mean(Orthogonal), color='blue', linestyle='--', label='Average estimated effect')
    axs[1].axvline(x=np.mean(Naive), color='blue', linestyle='--', label='Average estimated effect')
    axs[0].axvline(x=10, color='red', linestyle='--', label='True Effect')
    axs[1].axvline(x=10, color='red', linestyle='--', label='True Effect')
    axs[0].title.set_text('Orthogonal estimation')
    axs[1].title.set_text('Naive estimation')
    axs[0].set_xlabel('Estimated Treatment Effect')
    axs[1].set_xlabel('Estimated Treatment Effect')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(f'Simulation with {B} iterations')
    plt.show()

    print(f'Time taken for {B} iteration simulation: {elapsed_time:.2f} seconds')
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p3_output_17_0.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    


    Time taken for 100 iteration simulation: 11.90 seconds
    


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p3_output_17_2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    


    Time taken for 1000 iteration simulation: 79.43 seconds
    


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p3_output_17_4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    


    Time taken for 10000 iteration simulation: 906.26 seconds
    

We can se that the orthogonal estimation yields on average a coeficient more centered on the true effect than the naive estimation. This method of estimation that utilizes the residuals of lasso regresion and helps reduce bias more efficiently than the naive estimation method. This is because it leverages the properties explained on the begining of this notebook rather than just controlling for a relevant set of covariates which may induce endogeneity.

We have used parallel computing because it is supposed to allow us to achieve better processing times. It can significantly lower the running time of computations due to its ability to distribute the workload across multiple processing units.

As an example, we can see how long it would have taken to run the simulation with 1000 iteration if we had not utilized parallel computing.

Without parallel computing


```python
np.random.seed(0)
B = 1000
Naive = np.zeros( B )
Orthogonal = np.zeros( B )

start_time = time.time()
for i in range( 0, B ):
    n = 100
    p = 100
    beta = ( 1 / (np.arange( 1, p + 1 ) ** 2 ) ).reshape( p , 1 )
    gamma = ( 1 / (np.arange( 1, p + 1 ) ** 2 ) ).reshape( p , 1 )

    mean = 0
    sd = 1
    X = np.random.normal( mean , sd, n * p ).reshape( n, p )

    D = ( X @ gamma ) + np.random.normal( mean , sd, n ).reshape( n, 1 )/4 # We reshape because in r when we sum a vecto with a matrix it sum by column
    
    # DGP 
    Y = 10*D + ( X @ beta ) + np.random.normal( mean , sd, n ).reshape( n, 1 )

    # single selection method
    r_lasso_estimation = hdmpy.rlasso( np.concatenate( ( D , X ) , axis  =  1 ) , Y , post = True ) # Regress main equation by lasso

    coef_array = r_lasso_estimation.est[ 'coefficients' ].iloc[ 2:, :].to_numpy()    # Get "X" coefficients 

    SX_IDs = np.where( coef_array != 0 )[0]

    # In case all X coefficients are zero, then regress Y on D
    if sum(SX_IDs) == 0 : 
        Naive[ i ] = sm.OLS( Y , sm.add_constant(D) ).fit().summary2().tables[1].round(3).iloc[ 1, 0 ] 

    # Otherwise, then regress Y on X and D (but only in the selected coefficients)
    elif sum( SX_IDs ) > 0 :
        X_D = np.concatenate( ( D, X[:, SX_IDs ] ) , axis = 1 )
        Naive[ i ] = sm.OLS( Y , sm.add_constant( X_D ) ).fit().summary2().tables[1].round(3).iloc[ 1, 0]

    # In both cases we save D coefficient
        
    # Regress residuals. 
    resY = hdmpy.rlasso( X , Y , post = False ).est[ 'residuals' ]
    resD = hdmpy.rlasso( X , D , post = False ).est[ 'residuals' ]
    Orthogonal[ i ] = sm.OLS( resY , sm.add_constant( resD ) ).fit().summary2().tables[1].round(3).iloc[ 1, 0]
end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f'Time taken for {B} iteration simulation without multiprocessing: {elapsed_time:.2f} seconds')
```

    Time taken for 1000 iteration simulation without multiprocessing: 488.92 seconds
    

We can see that if we were to not use parallel computing, the processing time would be higher. It took 488 seconds to achieve what we achieved in 79.

### 2.2. Double Lasso - Using School data


```python
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
```

#### 2.2.1. Preprocessing data


```python
# Read csv file
df = pd.read_csv('./data/bruhn2016.csv', delimiter=',')
```


```python
df.head()
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
      <th>outcome.test.score</th>
      <th>treatment</th>
      <th>school</th>
      <th>is.female</th>
      <th>mother.attended.secondary.school</th>
      <th>father.attened.secondary.school</th>
      <th>failed.at.least.one.school.year</th>
      <th>family.receives.cash.transfer</th>
      <th>has.computer.with.internet.at.home</th>
      <th>is.unemployed</th>
      <th>has.some.form.of.income</th>
      <th>saves.money.for.future.purchases</th>
      <th>intention.to.save.index</th>
      <th>makes.list.of.expenses.every.month</th>
      <th>negotiates.prices.or.payment.methods</th>
      <th>financial.autonomy.index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47.367374</td>
      <td>0</td>
      <td>17018390</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58.176758</td>
      <td>1</td>
      <td>33002614</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>41.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56.671661</td>
      <td>1</td>
      <td>35002914</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>48.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29.079376</td>
      <td>0</td>
      <td>35908915</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49.563534</td>
      <td>1</td>
      <td>33047324</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>31.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop missing values, we lose 5077 values (from 17299 to 12222 rows)
df.dropna(axis=0, inplace=True)
df.reset_index(inplace=True ,drop=True)
```


```python
df.columns
```




    Index(['outcome.test.score', 'treatment', 'school', 'is.female',
           'mother.attended.secondary.school', 'father.attened.secondary.school',
           'failed.at.least.one.school.year', 'family.receives.cash.transfer',
           'has.computer.with.internet.at.home', 'is.unemployed',
           'has.some.form.of.income', 'saves.money.for.future.purchases',
           'intention.to.save.index', 'makes.list.of.expenses.every.month',
           'negotiates.prices.or.payment.methods', 'financial.autonomy.index'],
          dtype='object')




```python
dependent_vars = ['outcome.test.score', 'intention.to.save.index', 'negotiates.prices.or.payment.methods', 'has.some.form.of.income', 'makes.list.of.expenses.every.month', 'financial.autonomy.index', 'saves.money.for.future.purchases', 'is.unemployed']
```

For Lasso regressions, we split the data into train and test data, and standarize the covariates matrix


```python
# Train test split
X = df.drop(dependent_vars, axis = 1)
y = df[dependent_vars]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```


```python
T_train = X_train['treatment']
T_test = X_test['treatment']

X_train = X_train.drop(['treatment'], axis = 1)
X_test = X_test.drop(['treatment'], axis = 1)

```


```python
# Standarize X data
scale = StandardScaler()

X_train_scaled = pd.DataFrame(scale.fit_transform(X_train), index=X_train.index)
X_test_scaled = pd.DataFrame(scale.transform(X_test), index=X_test.index)
```


```python
X_scaled = pd.concat([X_train_scaled, X_test_scaled]).sort_index()
T = pd.concat([T_train, T_test]).sort_index()
```

#### 2.2.2. Regressions

##### a. OLS

From 1 - 3 regression: measures treatment impact on **student financial proficiency**

From 4 - 6 regression: measures treatment impact on **student savings behavior and attitudes**

From 7 - 9 regression: measures treatment impact on **student money management behavior and attitudes**

From 10 - 12 regression: measures treatment impact on **student entrepreneurship and work outcomes**


```python
# Rgeressions with "Student Financial Proficiency" as dependet variable
ols_score_1      = sm.OLS.from_formula('Q("outcome.test.score") ~ treatment', data=df).fit()
ols_score_2      = sm.OLS.from_formula('Q("outcome.test.score") ~ treatment + school + Q("failed.at.least.one.school.year")', data=df).fit()
ols_score_3      = sm.OLS.from_formula('Q("outcome.test.score") ~ treatment + school + Q("failed.at.least.one.school.year") + Q("is.female") + Q("mother.attended.secondary.school") + Q("father.attened.secondary.school") + Q("family.receives.cash.transfer") + Q("has.computer.with.internet.at.home")', data=df).fit()

# Rgeressions with "Intention to save index" as dependet variable
ols_saving_1     = sm.OLS.from_formula('Q("intention.to.save.index") ~ treatment', data=df).fit()
ols_saving_2     = sm.OLS.from_formula('Q("intention.to.save.index") ~ treatment + school + Q("failed.at.least.one.school.year")', data=df).fit()
ols_saving_3     = sm.OLS.from_formula('Q("intention.to.save.index") ~ treatment + school + Q("failed.at.least.one.school.year") + Q("is.female") + Q("mother.attended.secondary.school") + Q("father.attened.secondary.school") + Q("family.receives.cash.transfer") + Q("has.computer.with.internet.at.home")', data=df).fit()

# Rgeressions with "Negotiates prices or payment methods" as dependet variable
ols_negotiates_1 = sm.OLS.from_formula('Q("negotiates.prices.or.payment.methods") ~ treatment', data=df).fit()
ols_negotiates_2 = sm.OLS.from_formula('Q("negotiates.prices.or.payment.methods") ~ treatment + school + Q("failed.at.least.one.school.year")', data=df).fit()
ols_negotiates_3 = sm.OLS.from_formula('Q("negotiates.prices.or.payment.methods") ~ treatment + school + Q("failed.at.least.one.school.year") + Q("is.female") + Q("mother.attended.secondary.school") + Q("father.attened.secondary.school") + Q("family.receives.cash.transfer") + Q("has.computer.with.internet.at.home")', data=df).fit()

# Rgeressions with "Has some form of income" as dependet variable
ols_manage_1     = sm.OLS.from_formula('Q("has.some.form.of.income") ~ treatment', data=df).fit()
ols_manage_2     = sm.OLS.from_formula('Q("has.some.form.of.income") ~ treatment + school + Q("failed.at.least.one.school.year")', data=df).fit()
ols_manage_3     = sm.OLS.from_formula('Q("has.some.form.of.income") ~ treatment + school + Q("failed.at.least.one.school.year") + Q("is.female") + Q("mother.attended.secondary.school") + Q("father.attened.secondary.school") + Q("family.receives.cash.transfer") + Q("has.computer.with.internet.at.home")', data=df).fit()

# Show parameters in table
st = Stargazer([ols_score_1, ols_score_2, ols_score_3, ols_saving_1, ols_saving_2, ols_saving_3, ols_negotiates_1, ols_negotiates_2, ols_negotiates_3, ols_manage_1, ols_manage_2, ols_manage_3])
st.custom_columns(["Dependent var 1: Student Financial Proficiency", "Dependent var 2: Intention to save index", "Dependent var 3: Negotiates prices or payment methods", "Dependent var 4: Has some form of income"], [3, 3, 3, 3])
st.rename_covariates({'Q("failed.at.least.one.school.year")': 'Failed at least one school year', 'Q("is.female")': 'Female', 'Q("father.attened.secondary.school")': 'Father attended secondary school', 'Q("Family.receives.cash.transfer")': 'Family receives cash transfer', 'Q("has.computer.with.internet.at.home")': 'Has computer with internet at home'})
st
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    




<table style="text-align:center"><tr><td colspan="13" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><tr><td></td><td colspan="3">Dependent var 1: Student Financial Proficiency</td><td colspan="3">Dependent var 2: Intention to save index</td><td colspan="3">Dependent var 3: Negotiates prices or payment methods</td><td colspan="3">Dependent var 4: Has some form of income</td></tr><tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td><td>(3)</td><td>(4)</td><td>(5)</td><td>(6)</td><td>(7)</td><td>(8)</td><td>(9)</td><td>(10)</td><td>(11)</td><td>(12)</td></tr>
<tr><td colspan="13" style="border-bottom: 1px solid black"></td></tr>

<tr><td style="text-align:left">Intercept</td><td>57.591<sup>***</sup></td><td>59.377<sup>***</sup></td><td>58.860<sup>***</sup></td><td>49.016<sup>***</sup></td><td>46.725<sup>***</sup></td><td>46.603<sup>***</sup></td><td>0.763<sup>***</sup></td><td>0.856<sup>***</sup></td><td>0.855<sup>***</sup></td><td>0.639<sup>***</sup></td><td>0.534<sup>***</sup></td><td>0.609<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.187)</td><td>(0.556)</td><td>(0.675)</td><td>(0.240)</td><td>(0.728)</td><td>(0.890)</td><td>(0.006)</td><td>(0.017)</td><td>(0.020)</td><td>(0.006)</td><td>(0.019)</td><td>(0.023)</td></tr>
<tr><td style="text-align:left">Failed at least one school year</td><td></td><td>-7.218<sup>***</sup></td><td>-6.652<sup>***</sup></td><td></td><td>-3.614<sup>***</sup></td><td>-3.315<sup>***</sup></td><td></td><td>0.024<sup>***</sup></td><td>0.013<sup></sup></td><td></td><td>0.005<sup></sup></td><td>0.006<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.288)</td><td>(0.289)</td><td></td><td>(0.377)</td><td>(0.381)</td><td></td><td>(0.009)</td><td>(0.009)</td><td></td><td>(0.010)</td><td>(0.010)</td></tr>
<tr><td style="text-align:left">Q("family.receives.cash.transfer")</td><td></td><td></td><td>-1.837<sup>***</sup></td><td></td><td></td><td>-1.189<sup>***</sup></td><td></td><td></td><td>0.028<sup>***</sup></td><td></td><td></td><td>-0.027<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td>(0.283)</td><td></td><td></td><td>(0.374)</td><td></td><td></td><td>(0.009)</td><td></td><td></td><td>(0.010)</td></tr>
<tr><td style="text-align:left">Father attended secondary school</td><td></td><td></td><td>0.875<sup>***</sup></td><td></td><td></td><td>-0.213<sup></sup></td><td></td><td></td><td>-0.012<sup></sup></td><td></td><td></td><td>0.021<sup>**</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td>(0.298)</td><td></td><td></td><td>(0.392)</td><td></td><td></td><td>(0.009)</td><td></td><td></td><td>(0.010)</td></tr>
<tr><td style="text-align:left">Has computer with internet at home</td><td></td><td></td><td>-0.505<sup>*</sup></td><td></td><td></td><td>-0.276<sup></sup></td><td></td><td></td><td>0.024<sup>***</sup></td><td></td><td></td><td>-0.035<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td>(0.281)</td><td></td><td></td><td>(0.371)</td><td></td><td></td><td>(0.009)</td><td></td><td></td><td>(0.010)</td></tr>
<tr><td style="text-align:left">Female</td><td></td><td></td><td>2.943<sup>***</sup></td><td></td><td></td><td>1.403<sup>***</sup></td><td></td><td></td><td>-0.069<sup>***</sup></td><td></td><td></td><td>-0.051<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td>(0.257)</td><td></td><td></td><td>(0.339)</td><td></td><td></td><td>(0.008)</td><td></td><td></td><td>(0.009)</td></tr>
<tr><td style="text-align:left">Q("mother.attended.secondary.school")</td><td></td><td></td><td>0.968<sup>***</sup></td><td></td><td></td><td>1.192<sup>***</sup></td><td></td><td></td><td>0.001<sup></sup></td><td></td><td></td><td>0.013<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td>(0.295)</td><td></td><td></td><td>(0.388)</td><td></td><td></td><td>(0.009)</td><td></td><td></td><td>(0.010)</td></tr>
<tr><td style="text-align:left">school</td><td></td><td>0.000<sup></sup></td><td>-0.000<sup>**</sup></td><td></td><td>0.000<sup>***</sup></td><td>0.000<sup>***</sup></td><td></td><td>-0.000<sup>***</sup></td><td>-0.000<sup>***</sup></td><td></td><td>0.000<sup>***</sup></td><td>0.000<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.000)</td><td>(0.000)</td><td></td><td>(0.000)</td><td>(0.000)</td><td></td><td>(0.000)</td><td>(0.000)</td><td></td><td>(0.000)</td><td>(0.000)</td></tr>
<tr><td style="text-align:left">treatment</td><td>4.216<sup>***</sup></td><td>4.392<sup>***</sup></td><td>4.325<sup>***</sup></td><td>-0.070<sup></sup></td><td>-0.005<sup></sup></td><td>-0.032<sup></sup></td><td>0.001<sup></sup></td><td>0.001<sup></sup></td><td>0.003<sup></sup></td><td>0.017<sup>**</sup></td><td>0.016<sup>*</sup></td><td>0.018<sup>**</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.261)</td><td>(0.255)</td><td>(0.253)</td><td>(0.335)</td><td>(0.334)</td><td>(0.333)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.009)</td><td>(0.009)</td><td>(0.009)</td></tr>

<td colspan="13" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align: left">Observations</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td><td>12222</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.021</td><td>0.069</td><td>0.086</td><td>0.000</td><td>0.009</td><td>0.013</td><td>0.000</td><td>0.004</td><td>0.012</td><td>0.000</td><td>0.003</td><td>0.011</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.021</td><td>0.068</td><td>0.085</td><td>-0.000</td><td>0.009</td><td>0.012</td><td>-0.000</td><td>0.004</td><td>0.012</td><td>0.000</td><td>0.003</td><td>0.010</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>14.432 (df=12220)</td><td>14.076 (df=12218)</td><td>13.949 (df=12213)</td><td>18.506 (df=12220)</td><td>18.421 (df=12218)</td><td>18.393 (df=12213)</td><td>0.425 (df=12220)</td><td>0.424 (df=12218)</td><td>0.423 (df=12213)</td><td>0.478 (df=12220)</td><td>0.477 (df=12218)</td><td>0.475 (df=12213)</td></tr><tr><td style="text-align: left">F Statistic</td><td>260.547<sup>***</sup> (df=1; 12220)</td><td>300.463<sup>***</sup> (df=3; 12218)</td><td>143.315<sup>***</sup> (df=8; 12213)</td><td>0.043<sup></sup> (df=1; 12220)</td><td>38.533<sup>***</sup> (df=3; 12218)</td><td>19.812<sup>***</sup> (df=8; 12213)</td><td>0.018<sup></sup> (df=1; 12220)</td><td>16.352<sup>***</sup> (df=3; 12218)</td><td>18.841<sup>***</sup> (df=8; 12213)</td><td>3.843<sup>**</sup> (df=1; 12220)</td><td>12.839<sup>***</sup> (df=3; 12218)</td><td>16.603<sup>***</sup> (df=8; 12213)</td></tr>
<tr><td colspan="13" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td><td colspan="12" style="text-align: right"><sup>*</sup>p&lt;0.1; <sup>**</sup>p&lt;0.05; <sup>***</sup>p&lt;0.01</td></tr></table>




```python
# Save the ITT beta and the confidence intervals
beta_OLS = ols_score_3.params['treatment']
conf_int_OLS = ols_score_3.conf_int().loc['treatment']
```

##### b. Double Lasso using cross validation

We use the first dependent variable (Student Financial Proficiency)

Step 1: We ran Lasso regression of _Y_ (student financial proficiency) on _X_, and _T_ (treatment) on _X_


```python
lasso_CV_yX = LassoCV(alphas = np.arange(0.0001, 0.5, 0.001), cv = 10, max_iter = 5000)
lasso_CV_yX.fit(X_train_scaled, y_train['outcome.test.score'])

lasso_CV_lambda = lasso_CV_yX.alpha_
print(f"Mejor lambda: {lasso_CV_lambda:.4f}")
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Mejor lambda: 0.0001
    


```python
# Estimate y predictions with all X
y_pred_yX = lasso_CV_yX.predict(X_scaled)
```


```python
lasso_CV_TX = LassoCV(alphas = np.arange(0.0001, 0.5, 0.001), cv = 10, max_iter = 5000)
lasso_CV_TX.fit(X_train_scaled, T_train)
y_pred = lasso_CV_TX.predict(X_test_scaled)

lasso_CV_lambda = lasso_CV_TX.alpha_
print(f"Mejor lambda: {lasso_CV_lambda:.4f}")
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Mejor lambda: 0.0011
    


```python
# Estimate T predictions with all X
y_pred_TX = lasso_CV_TX.predict(X_scaled)
```

Step 2: Obtain the resulting residuals


```python
res_yX = y['outcome.test.score'] - y_pred_yX
res_TX = T - y_pred_TX
```

Step 3: We run the least squares of res_yX on res_TX


```python
ols_score_b = sm.OLS.from_formula('res_yX ~ res_TX', data=df).fit()

# Show parameters in table
st = Stargazer([ols_score_b])
st
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    




<table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td colspan="1"><em>Dependent variable: res_yX</em></td></tr><tr><td style="text-align:left"></td><tr><td style="text-align:left"></td><td>(1)</td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr>

<tr><td style="text-align:left">Intercept</td><td>0.033<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.126)</td></tr>
<tr><td style="text-align:left">res_TX</td><td>4.324<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.253)</td></tr>

<td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align: left">Observations</td><td>12222</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.023</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.023</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>13.945 (df=12220)</td></tr><tr><td style="text-align: left">F Statistic</td><td>292.956<sup>***</sup> (df=1; 12220)</td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td><td colspan="1" style="text-align: right"><sup>*</sup>p&lt;0.1; <sup>**</sup>p&lt;0.05; <sup>***</sup>p&lt;0.01</td></tr></table>




```python
# Save the ITT beta and the confidence intervals
beta_DL_CV = ols_score_b.params['res_TX']
conf_int_DL_CV = ols_score_b.conf_int().loc['res_TX']
```

##### c. Double Lasso using theoretical lambda


```python
# !pip install multiprocess
# !pip install pyreadr
# !git clone https://github.com/maxhuppertz/hdmpy.git
```


```python
import sys
sys.path.insert(1, "./hdmpy")
```


```python
# We wrap the package so that it has the familiar sklearn API
import hdmpy
from sklearn.base import BaseEstimator, clone

class RLasso(BaseEstimator):

    def __init__(self, *, post=True):
        self.post = post

    def fit(self, X, y):
        self.rlasso_ = hdmpy.rlasso(X, y, post=self.post)
        return self

    def predict(self, X):
        return np.array(X) @ np.array(self.rlasso_.est['beta']).flatten() + np.array(self.rlasso_.est['intercept'])

    def nsel(self):
        return sum(abs(np.array(self.rlasso_.est['beta']).flatten()>0))

lasso_model = lambda: RLasso(post=False)
```

Step 1:


```python
# Estimate y predictions with all X
y_pred_yX = lasso_model().fit(X_scaled, y['outcome.test.score']).predict(X_scaled)
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    


```python
# Estimate T predictions with all X
y_pred_TX = lasso_model().fit(X_scaled, T).predict(X_scaled)
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    

Step 2:


```python
res_yX = y['outcome.test.score'] - y_pred_yX
res_TX = T - y_pred_TX
```

Step 3:


```python
lasso_hdm_score = sm.OLS.from_formula('res_yX ~ res_TX', data=df).fit()

# Show parameters in table
st = Stargazer([lasso_hdm_score])
st
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    




<table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td colspan="1"><em>Dependent variable: res_yX</em></td></tr><tr><td style="text-align:left"></td><tr><td style="text-align:left"></td><td>(1)</td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr>

<tr><td style="text-align:left">Intercept</td><td>0.000<sup></sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.126)</td></tr>
<tr><td style="text-align:left">res_TX</td><td>4.316<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.253)</td></tr>

<td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align: left">Observations</td><td>12222</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.023</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.023</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>13.953 (df=12220)</td></tr><tr><td style="text-align: left">F Statistic</td><td>291.837<sup>***</sup> (df=1; 12220)</td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td><td colspan="1" style="text-align: right"><sup>*</sup>p&lt;0.1; <sup>**</sup>p&lt;0.05; <sup>***</sup>p&lt;0.01</td></tr></table>




```python
# Save the ITT beta and the confidence intervals
beta_DL_theo = lasso_hdm_score.params['res_TX']
conf_int_DL_theo = lasso_hdm_score.conf_int().loc['res_TX']
```

##### d. Double Lasso using partialling out method


```python
rlassoEffect = hdmpy.rlassoEffect(X_scaled, y['outcome.test.score'], T, method='partialling out')
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    


```python
rlassoEffect
```




    {'alpha': 4.313441,
     'se': array([0.25271166]),
     't': array([17.06862565]),
     'pval': array([2.54111627e-65]),
     'coefficients': 4.313441,
     'coefficient': 4.313441,
     'coefficients_reg':                      0
     (Intercept)  59.769260
     x0            0.000000
     x1            1.511205
     x2            0.529423
     x3            0.461196
     x4           -2.878027
     x5           -0.857569
     x6            0.000000,
     'selection_index': array([[False],
            [ True],
            [ True],
            [ True],
            [ True],
            [ True],
            [False]]),
     'residuals': {'epsilon': array([[-10.04200277],
             [-31.30841071],
             [-15.13769394],
             ...,
             [-17.22383794],
             [ -3.93047339],
             [ -4.88461742]]),
      'v': array([[ 0.48682705],
             [-0.513173  ],
             [ 0.48682705],
             ...,
             [ 0.48682705],
             [-0.513173  ],
             [-0.513173  ]], dtype=float32)},
     'samplesize': 12222}




```python
beta_part_out = rlassoEffect['coefficient']
```


```python
critical_value = 1.96  # For 95% confidence level

conf_int_part_out = [beta_part_out - critical_value * rlassoEffect['se'], \
                     beta_part_out + critical_value * rlassoEffect['se']]
```

#### Results

We found that the intention to treat effect (ITT) is very similar estimating with all 4 models (aproximately 4.3, with 95% of confidence). This could be because the ratio between the parameters and the number of observations p/n is small (8/12222 = 0.00065455735). In other words, we are not dealing with high dimensional data and the models from b. to d. will outperform the OLS when we are in the opposite scenario. In conclusion, we can say that the OLS model estimates the ITT just as good as the other models.


```python
# Plotting the effect size with confidence intervals
plt.figure(figsize=(8, 6))
plt.errorbar('OLS', beta_OLS, yerr=np.array([beta_OLS - conf_int_OLS[0], conf_int_OLS[1] - beta_OLS]).reshape(2, 1), 
             fmt='o', color='black', capsize=5)
plt.errorbar('Double Lasso with CV', beta_DL_CV, yerr=np.array([beta_DL_CV - conf_int_DL_CV[0], conf_int_DL_CV[1] - beta_DL_CV]).reshape(2, 1), 
             fmt='o', color='black', capsize=5)
plt.errorbar('Double Lasso with theoretical lambda', beta_DL_theo, yerr=np.array([beta_DL_theo - conf_int_DL_theo[0], conf_int_DL_theo[1] - beta_DL_theo]).reshape(2, 1), 
             fmt='o', color='black', capsize=5)
plt.errorbar('Double Lasso with partialling out', beta_part_out, yerr=np.array([beta_part_out - conf_int_part_out[0], conf_int_part_out[1] - beta_part_out]).reshape(2, 1), 
             fmt='o', color='black', capsize=5)
plt.title('Intention to treat effect on Student Financial Proficiency')
plt.ylabel('Beta and cofidence interval')
plt.xticks(rotation=45)

plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p3_output_72_0.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    


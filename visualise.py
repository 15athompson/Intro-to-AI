# To visualize this association, we can create a stacked bar plot:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# create a sample dataframe
df = pd.DataFrame({
    'Occupation': np.random.choice(['employed', 'self-employed'], size=100),
    'Travel': np.random.choice(['car', 'bus', 'other'], size=100),
    'Education': np.random.choice(['high', 'uni'], size=100),
    'Residence': np.random.choice(['small', 'big'], size=100)
})


# create cross-tabulation of Occupation and Travel
occupation_transport = pd.crosstab(df['Occupation'], df['Travel'], normalize='index')

# plot stacked bar chart
occupation_transport.plot(kind='bar', stacked=True)
plt.title('Preferred Mode of Transportation by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Travel')
plt.show()

# create cross-tabulation of Education and Travel
education_transport = pd.crosstab(df['Education'], df['Travel'], normalize='index')


#  distribution of preferred mode of transportation across different education levels:

edu_counts = pd.crosstab(df["Education"], df["Travel"])
edu_counts.plot(kind="bar", stacked=True)
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.title("Preferred Mode of Transportation by Education Level")
plt.show()


# test for a significant association between education level and preferred mode of transportation:

chi2_contingency(edu_counts)

# create cross-tabulation of Education and Travel
occupation_transport = pd.crosstab(df['Education'], df['Travel'], normalize='index')

# distribution of preferred mode of transportation across different occupations:

occ_counts = pd.crosstab(df["Occupation"], df["Travel"])
occ_counts.plot(kind="bar", stacked=True)
plt.xlabel("Occupation")
plt.ylabel("Count")
plt.title("Preferred Mode of Transportation by Occupation")
plt.show()

#  test for a significant association between occupation and preferred mode of transportation

chi2_contingency(occ_counts)


# create cross-tabulation of Education and Travel
residence_transport = pd.crosstab(df['Residence'], df['Travel'], normalize='index')


#distribution of preferred mode of transportation across different city sizes:

res_counts = pd.crosstab(df["Residence"], df["Travel"])
res_counts.plot(kind="bar", stacked=True)
plt.xlabel("City Size")
plt.ylabel("Count")
plt.title("Preferred Mode of Transportation by City Size")
plt.show()

#  test for a significant association between residence and preferred mode of transportation

chi2_contingency(res_counts)

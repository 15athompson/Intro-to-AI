import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("Travel_data.csv")

# Print the first five rows of the data
print(data.head())

# Explore the data
print(data.describe())

# Investigate the relationship between the preferred mode of transportation and some socioeconomic and demographic variables
sns.countplot(x="Favourite means of transport", hue="Sex", data=data)
plt.show()

sns.countplot(x="Favourite means of transport", hue="Age", data=data)
plt.show()

sns.countplot(x="Favourite means of transport", hue="Education", data=data)
plt.show()

sns.countplot(x="Favourite means of transport", hue="Occupation", data=data)
plt.show()

sns.countplot(x="Favourite means of transport", hue="Residence", data=data)
plt.show()

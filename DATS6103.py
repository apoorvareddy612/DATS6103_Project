#%%
#This is our main code file
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#%%
data = pd.read_csv('cleaned_5250.csv')
print(data.head())
count_data = data.groupby(['discovery_year', 'detection_method']).size().unstack(fill_value=0)

# Plot the stacked bar plot
ax = count_data.plot(kind='bar', stacked=True, colormap='viridis')

# Customize the plot
ax.set_title('NASA Exoplanet Discovery Year and Detection Method')
ax.set_xlabel('Discovery Year')
ax.set_ylabel('Number of Exoplanets')
ax.legend(title='Detection Method', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

count_data = data.groupby(['planet_type', 'detection_method']).size().unstack(fill_value=0)

# Plot the stacked bar plot
ax = count_data.plot(kind='bar', stacked=True, colormap='magma')

# Customize the plot
ax.set_title('NASA Exoplanet Discovery Year and Detection Method')
ax.set_xlabel('Planet Type')
ax.set_ylabel('Number of Exoplanets')
ax.legend(title='Detection Method', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

#Orbital Period
data['orbital_period'].value_counts()
data['Orbital_Period'] = ''
for i in range (len(data)):
    if data['orbital_period'][i] < 1:
        data['Orbital_Period'][i] = '<1'
    elif data['Orbital_Period'][i] == 1:
        data['Orbital_Period'][i] = '1'
    else:
        data['Orbital_Period'][i] = '>1'

sns.boxplot(x='discovery_year', y='Orbital_Period', data=data)
plt.title('Discovery Year vs Orbital Period of Exoplanets')
plt.xlabel('Discovery Year')
plt.ylabel('Orbital Period (years)')
plt.show()

# %%
plt.figure(figsize=(30, 10))
sns.countplot(x='detection_method', data=data, palette='viridis')
# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance', y='stellar_magnitude', data=data, alpha=0.7)
plt.title('Relationship Between Distance and Stellar Magnitude in Exoplanet Discovery')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Stellar Magnitude')
plt.grid(True)
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.scatter(data['stellar_magnitude'],data['discovery_year'], alpha=0.5)
plt.title('Stellar Magnitude vs Exoplanet Discovery Year')
plt.xlabel('Stellar Magnitude')
plt.ylabel('Discovery Year')
plt.grid(True)
plt.show()

#%%
## Correlation Matrix

numerical_columns = data.select_dtypes(include = ['float64','int64']).columns

correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Exoplanet Parameters")
plt.show()

#%%
#Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('cleaned.csv')

# Encoding the categorical target variable
label_encoder = LabelEncoder()
data['planet_type_encoded'] = label_encoder.fit_transform(data['planet_type'])

# Selecting features and target variable
features = ['mass_multiplier', 'radius_multiplier']
target = 'planet_type_encoded'

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Creating and training the random forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

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

interpretation = '''As we can see from the above plot, eccentricity and orbital period have  a positive correlation. When eccentricity increases, the orbital period also increases in a linear fashion and vice versa.
Also, a very strong positive correlation can be seen between the variables orbital radius and orbital period. This implies that the planets farther away from their host star take longer to complete one  orbit.'''

print(interpretation)

#%%
#Chi square test to check if there a significant association between planet type and detection method
from scipy.stats import chi2_contingency

Hypothesis = '''Null Hypothesis(H0) : There is a significant association between planet type and detection methods.
Alternative Hypothesis(Ha) : There is no significant association between planet types and detection methods'''

print(Hypothesis)

# Contingency table (cross-tabulation) of planet_type and detection_method
contingency_table = pd.crosstab(data['planet_type'], data['detection_method'])

# Chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Print the results
print("Chi-square value:", chi2)
print("p-value:", p)

# Interpret the results
alpha = 0.05  # Set your significance level
print("\nSignificance test:")
if p < alpha:
    print("There is a significant association between planet type and detection method.")
else:
    print("There is no significant association between planet type and detection method.")

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

# %%

#%%
#Relationship Between Planet Type and Orbital Characteristics:    
#Do certain planet types have distinct orbital periods, eccentricities, or orbital radii?
data = pd.read_csv('cleaned.csv')

# Grouping by planet type and calculating mean orbital characteristics
orbital_characteristics = data.groupby('planet_type').agg({
    'orbital_period': 'mean',
    'eccentricity': 'mean',
    'orbital_radius': 'mean'
}).reset_index()

# Plotting the relationship between planet type and orbital characteristics
plt.figure(figsize=(12, 6))

# Orbital Period vs Planet Type
plt.subplot(131)
sns.barplot(x='planet_type', y='orbital_period', data=orbital_characteristics)
plt.title('Mean Orbital Period by Planet Type')
plt.xticks(rotation=45)

# Eccentricity vs Planet Type
plt.subplot(132)
sns.barplot(x='planet_type', y='eccentricity', data=orbital_characteristics)
plt.title('Mean Eccentricity by Planet Type')
plt.xticks(rotation=45)

# Orbital Radius vs Planet Type
plt.subplot(133)
sns.barplot(x='planet_type', y='orbital_radius', data=orbital_characteristics)
plt.title('Mean Orbital Radius by Planet Type')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
#Are there correlations between planet types and specific detection methods?

# Creating a cross-tabulation between planet types and detection methods
cross_tab = pd.crosstab(data['planet_type'], data['detection_method'])

# Plotting a heatmap to visualize correlations
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='d')
plt.title('Correlation between Planet Types and Detection Methods')
plt.xlabel('Detection Method')
plt.ylabel('Planet Type')
plt.show()

# %%
# %%
#What is the relationship between the distance of exoplanets from their host stars and their planetary characteristics, such as mass and radius?

# Scatter plot for distance vs mass
plt.figure(figsize=(10, 6))
plt.scatter(data['orbital_radius'], data['mass_multiplier'], alpha=0.5)
plt.title('Orbital Radius vs Mass of Exoplanets')
plt.xlabel('Orbital Radius (AU)')
plt.ylabel('Mass Multiplier (Relative to Jupiter)')
plt.grid(True)
plt.show()

# Scatter plot for distance vs radius
plt.figure(figsize=(10, 6))
plt.scatter(data['orbital_radius'], data['radius_multiplier'], alpha=0.5)
plt.title('Orbital Radius vs Radius of Exoplanets')
plt.xlabel('Orbital Radius (AU)')
plt.ylabel('Radius Multiplier (Relative to Jupiter)')
plt.grid(True)
plt.show()
# %%
#KNN model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Selecting features and target variable
features = ['mass_multiplier', 'radius_multiplier']
target = 'planet_type'  

# Dropping rows with missing values in the selected features and target
selected_data = data[features + [target]].dropna()

# Splitting the data into training and testing sets
X = selected_data[features]  
y = selected_data[target]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the KNN model
k = 5  
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Making predictions
y_pred = knn.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# %%

#For extracting temporal features from the 'discovery_year' column, we can extract month and season information
#Extracting month or season from the 'discovery_year' column might reveal patterns in discovery frequency across different times of the year, aiding in understanding any seasonal patterns or influences on exoplanet discoveries.

# Convert 'discovery_year' to datetime format
data['discovery_year'] = pd.to_datetime(data['discovery_year'], format='%Y')

# Extracting month from 'discovery_year'
data['discovery_month'] = data['discovery_year'].dt.month

# Extracting season from 'discovery_year'
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

data['discovery_season'] = data['discovery_month'].apply(get_season)


#%%
#Temporal Trends in Discoveries:
#How has the rate of exoplanet discoveries evolved over the years? Are there any significant spikes or declines?
#Visualizing Discoveries Over Time: Analyzing discoveries across different years helps identify trends, spikes, or declines in exoplanet discoveries. It reveals if there are periods of increased or decreased discovery rates.

# Convert 'discovery_year' to datetime format and extract the year
data['discovery_year'] = pd.to_datetime(data['discovery_year']).dt.year

# Count the number of discoveries per year
discoveries_per_year = data['discovery_year'].value_counts().sort_index()

# Plotting the number of discoveries over the years
plt.figure(figsize=(10, 6))
discoveries_per_year.plot(kind='line', marker='o')
plt.title('Number of Exoplanet Discoveries Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Discoveries')
plt.grid(True)
plt.show()
# %%
#%%
#Have certain detection methods become more prevalent or effective over time?
# Grouping by year and detection method to count occurrences
method_counts_over_time = data.groupby(['discovery_year', 'detection_method']).size().unstack(fill_value=0)

# Plotting the prevalence of detection methods over time
plt.figure(figsize=(12, 8))
method_counts_over_time.plot(kind='area', stacked=True)
plt.title('Prevalence of Detection Methods Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Discoveries')
plt.legend(title='Detection Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
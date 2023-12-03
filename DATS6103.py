#%%
#This is our main code file
## Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#%%
##Reading the file:
data = pd.read_csv('cleaned.csv')
print(data.head()) ##Printing the first five rows of dataset
#%%
##Count Plot to show the number of exoplanets discovered in each yeat from 1992-2023 using different detection methods
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
#%%
## Count Plot to show the total number of exoplanets discovered for each planet type  using different detection methods
count_data = data.groupby(['planet_type', 'detection_method']).size().unstack(fill_value=0)

# Plot the stacked bar plot
ax = count_data.plot(kind='bar', stacked=True, colormap='magma')

# Customize the plot
ax.set_title('NASA Exoplanet Planet Type and Detection Method')
ax.set_xlabel('Planet Type')
ax.set_ylabel('Number of Exoplanets')
ax.legend(title='Detection Method', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()
#%%
##Boxplot to o visualize the relationship between discovery year and orbital period
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
##Count plot to show the total number of exoplanets for each detection type.
plt.figure(figsize=(30, 10))
sns.countplot(x='detection_method', data=data, palette='viridis')
# %%
##Scatter plot to show the relationship between distance and stellar magnitude in exoplanet discovery
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance', y='stellar_magnitude', data=data, alpha=0.7)
plt.title('Relationship Between Distance and Stellar Magnitude in Exoplanet Discovery')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Stellar Magnitude')
plt.grid(True)
plt.show()
# %%
##Scatter Plot to show the relationship between stellar magnitude and discovery year.
plt.figure(figsize=(10, 6))
plt.scatter(data['stellar_magnitude'],data['discovery_year'], alpha=0.5)
plt.title('Stellar Magnitude vs Exoplanet Discovery Year')
plt.xlabel('Stellar Magnitude')
plt.ylabel('Discovery Year')
plt.grid(True)
plt.show()

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

#%%
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
#%%
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

#%%
## Correlation Matrix
numerical_columns = data.select_dtypes(include = ['float64','int64']).columns

correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Exoplanet Parameters")
plt.show()

#%%
#Examining the Impact of Discovery Year on Orbital Period Significance:
#
import statsmodels.api as sm

# Considering 'orbital_period' as the dependent variable and 'discovery_year' as the independent variable
X = data['discovery_year']
y = data['orbital_period']

# Adding a constant term to the independent variable
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Summary of the regression model
print(model.summary())


#%%
#Exploring Relationship between Distance and Planetary Characteristics:
# Scatter plot for distance vs mass and radius
plt.figure(figsize=(10, 6))
plt.scatter(data['distance'], data['mass'], alpha=0.5)
plt.title('Distance vs Mass of Exoplanets')
plt.xlabel('Distance from Host Star (parsecs)')
plt.ylabel('Mass Multiplier (Relative to Jupiter)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['distance'], data['radius'], alpha=0.5)
plt.title('Distance vs Radius of Exoplanets')
plt.xlabel('Distance from Host Star (parsecs)')
plt.ylabel('Radius Multiplier (Relative to Jupiter)')
plt.grid(True)
plt.show()


#%%



#%%
#Analyzing Relationships Among Orbital Characteristics and Host Stars:
# Pairplot to visualize relationships between orbital characteristics and host stars
sns.pairplot(data, vars=['orbital_radius', 'orbital_period', 'eccentricity', 'stellar_magnitude'])
plt.show()

# Correlation between features and correlation heatmap
correlation_matrix = data[['orbital_radius', 'orbital_period', 'eccentricity', 'stellar_magnitude']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Orbital Characteristics and Stellar Magnitude")
plt.show()


#%%

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

##Models
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
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# %%
#KNN model
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

#%%
## SVM model

# Create an SVM model with hyperparameter tuning
svm_model = SVC(kernel='rbf', C=100, gamma=10)  

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the  test set
predictions = svm_model.predict(X_test)

# Make predictions on the  test set
predictions = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report_result)


#%%
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# Create and train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gb_model.fit(X_train, y_train)

# Predict using the test set
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
classification_report_gb = classification_report(y_test, y_pred_gb)

print(f'Gradient Boosting Accuracy: {accuracy_gb:.2f}')
print('\nGradient Boosting Classification Report:')
print(classification_report_gb)




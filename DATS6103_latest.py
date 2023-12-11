#%%[markdown]
## PROJECT FALL 2023
#### DATS6103 'Introduction to Data Mining'
##### Team Members: Apoorva Reddy Bagepalli, Saniya, Pratiksha, Tulasi
##### Project description
#NASA(National Aeronautics and Space Administration) is researching and exploring the space for different planets, galaxies and other space wonders. In addition to searching planets, they are also investigating for the ones which can be suitable for the living which is beyond our solar system. Until now NASA discovered 5,550 exoplanets using different Detection methods.\
#\
# In our project we use the data consisting about different discovered exoplanets from 1992 to 2023. To analyse and study the patterns/trends of the exoplanets, which will help us to learn about them and their characteristics.\
#\
#We conduct EDA(Exploratory Data Analysis) and Predictive Modelling to learn about the features
#%%
#Main Code
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
from sklearn.model_selection import GridSearchCV


#%%
##Reading the file:
data = pd.read_csv('cleaned.csv')
print(data.head()) ##Printing the first five rows of dataset

# %%
#EDA
#Count plot to show the total number of exoplanets for each detection type.
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

#* Exoplanets with small stellar magnitude value and less distance from the Earth are brighter
#* But some exoplanets are bright even though they are far away from the Earth 
#* Most of the exoplanets discovered were near to the Earth and have smaller Stellar Magnitude value 

# %%
##Scatter Plot to show the relationship between stellar magnitude and discovery year.
plt.figure(figsize=(10, 6))
plt.scatter(data['stellar_magnitude'],data['discovery_year'], alpha=0.5)
plt.title('Stellar Magnitude vs Exoplanet Discovery Year')
plt.xlabel('Stellar Magnitude')
plt.ylabel('Discovery Year')
plt.grid(True)
plt.show()
#%%[markdown]
#* From the aobve plot we can interpret very easily that NASA was able to discover most of the exoplanets because of its smaller stellar magnitude values (<20)
#* Only one exoplanet was discovered which has more than 40 stellar magnitude value

#%%
#EDA
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
#%%[markdown]
#* Neptune-like exoplanets were mostly discovered compared to other type of exoplanets
#* Gas Giants xoplanets have been discovered by various detection methods

#%%
#EDA
## Correlation Matrix
numerical_columns = data.select_dtypes(include = ['float64','int64']).columns

correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Exoplanet Parameters")
plt.show()

#%%
#EDAs
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
#Start of Smart Questions
#%%
## 1. Count Plot to show the number of exoplanets discovered in each yeat from 1992-2023 using different detection methods
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

#%%[markdown]
#* There were 11 detection methods used since 1992 to 2023
#* Out of all the detection methods, Transit was used to discover more exoplanets than others (1453 exoplanets) in 2016 and other years too
#* Out of all the discovery years, 2016 had most exoplanet discoveries
#* Disk Kinematics was the least used detection method 

# %%
#For extracting temporal features from the 'discovery_year' column, we can extract month and season information
#Extracting month or season from the 'discovery_year' column might reveal patterns in discovery frequency across different times of the year, aiding in understanding any seasonal patterns or influences on exoplanet discoveries.

# # Convert 'discovery_year' to datetime format
data['discovery_year'] = pd.to_datetime(data['discovery_year'], format='%Y')

# Extracting month from 'discovery_year'
data['discovery_month'] = data['discovery_year'].dt.month

# # Extracting season from 'discovery_year'
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
# 2. How has the rate of exoplanet discoveries evolved over the years? Are there any significant spikes or declines?

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

#%%[markdown]
#* There is an overall exponential upward trend in the rate of exoplanet discoveries since the early 1990s, indicating rapidly accelerating detection capabilities. 
#* Two noticeable spikes occur in 2014 and 2016, suggesting breakthroughs in discovery techniques or observational power during these years. 
#* After the 2016 spike, the discovery rate declines somewhat but still remains higher than pre-2016 levels. This could indicate limitations in sustaining the suddenly increased discovery rates achieved in 2016. 


#%%
# 3. Boxplot to  visualize the relationship between discovery year and orbital period
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
#%%[markdown]
#* Exoplanets with >1 orbital period were easy to capture by the NASA and they found alot of them
#* Exoplanets with <1 orbital period were hard to find and there are limited discoveries
#* The range of exoplanets with >1 period is more than the exoplanets with <1 period
#%%
# 4. Analyzing Relationships Among Orbital Characteristics and Host Stars:
#Scatter plot to visualize relationships between orbital characteristics and host stars
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plots between pairs of variables
plt.figure(figsize=(10, 8))

plt.subplot(221)
sns.scatterplot(x='orbital_radius', y='orbital_period', data=data)
plt.title('Orbital Radius vs Orbital Period')

plt.subplot(222)
sns.scatterplot(x='orbital_radius', y='eccentricity', data=data)
plt.title('Orbital Radius vs Eccentricity')

plt.subplot(223)
sns.scatterplot(x='orbital_radius', y='stellar_magnitude', data=data)
plt.title('Orbital Radius vs Stellar Magnitude')

plt.subplot(224)
sns.scatterplot(x='orbital_period', y='eccentricity', data=data)
plt.title('Orbital Period vs Eccentricity')

plt.tight_layout()
plt.show()

#%%[markdown]
#* There is no clear correlation between an exoplanet's distance from its host star and its mass and radius relative to Jupiter as data points are broadly scattered.
#* Exoplanets span a wide range of masses (from less than Jupiter to over 10 times Jupiter's mass) across the full range of observed orbital distances shown.
#* While some clustering of lower mass planets occurs toward inner orbits, there are still exceptions. Broad spread suggests mass depends more strongly on other factors.
#* Upper limit of exoplanet radius reduces with distance, but smaller outliers exist.
#* Exoplanets display extensive diversity in radius across all observed orbital distances, indicating other influential factors in determining planet size.

# %%
#5. What is the relationship between the distance of exoplanets from their host stars and their planetary characteristics, such as mass and radius?

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

#%%[markdown]
#* No evident correlation between exoplanet distance from its star and its mass, radius relative to Jupiter; widely scattered data points.
#* Exoplanets showcase vast mass diversity across all observed orbital distances.
#* Lower mass planets predominantly cluster toward inner orbits, yet exceptions abound, indicating other influential factors.
#* Loose upper limit on exoplanet radius decreasing with distance from the host star, accompanied by numerous smaller outliers.
#* Substantial diversity in exoplanet radius across observed orbital distances suggests multiple factors influence planet size determination.

#%%
#6. Relationship Between Planet Type and Orbital Characteristics:    
#Do certain planet types have distinct orbital periods, eccentricities, or orbital radii?
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
#%%[markdown]
#* Planet Type vs Orbital Period:
#* Hot Jupiters have the shortest orbital periods on average, while cold/cool gas giants have longer orbital periods.
#* Rocky and cold gas planets span a wide range of possible periods.

#* Planet Type vs Eccentricity:
#* Hot Jupiters tend to have circular orbits with low eccentricity.
#* Rocky and cool planets display a large spread of eccentricities from circular to highly elliptical.

#* Planet Type vs Orbital Radius:
#* Hot Jupiters orbit extremely close to their host stars.
#* Cool/cold planets have wider orbital radii on average.
#* Orbit distances vary widely for rocky planets.


#%%[markdown]
#* Strong positive correlation: Mass increases with radius, reflecting internal density and composition relationships.
#* Moderate negative correlation: Orbital period decreases with orbital radius, likely due to gravitational effects.
#* Weak correlations: Eccentricity shows limited associations with other parameters, indicating diverse evolutionary factors.
#* Apart from mass-radius, weak correlations prevail, highlighting exoplanet diversity and complex planetary system dynamics.


 
#%%
# 7.Prediction
##Models: Random Forest

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
##Adding cross validation code to choose the best hyperparameters(C and gamma value)

# Create an SVM model with hyperparameter tuning
svm_model = SVC(kernel='rbf')

# Define the hyperparameters grid for tuning
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [10,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

# Instantiate GridSearchCV with the SVM model and parameters grid
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', verbose=1)

# Train the model with cross-validation
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator from the grid search
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Make predictions on the test set using the best estimator
predictions = best_estimator.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

print(f'Best Parameters: {best_params}')
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report_result)

#%%
#Accuracy Comparision of the models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Creating a dictionary to store classifier names and their instances
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', C=100, gamma=10),
    # 'Naive Bayes': GaussianNB(),
    # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
}

# Storing results
results = {'Algorithm': [], 'Accuracy': []}

# Evaluating each classifier
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results['Algorithm'].append(clf_name)
    results['Accuracy'].append(accuracy)

# Creating a DataFrame to display results
results_df = pd.DataFrame(results)

# Displaying the accuracy comparison
print(results_df)

# Plotting accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results_df['Algorithm'], results_df['Accuracy'], color='skyblue')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Classifiers')
plt.xticks(rotation=45)
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.show()







import pandas as pd
import numpy as np
import pgmpy as py
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, PC, BDeuScore, K2Score, BicScore
from pgmpy.estimators import StructureScore, ExhaustiveSearch
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from sklearn.metrics import f1_score
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

###############################################################################################
# 1. CREATE THE BAYESIAN NETWORK:



# Define the nodes in the network
model = BayesianModel([('Age', 'Education'), ('Sex', 'Education'), ('Education', 'Residence'), ('Education', 'Occupation'), ('Residence', 'Travel'), ('Occupation', 'Travel')])

# Draw the graph
pos = nx.kamada_kawai_layout(model)
nx.draw(model, pos, with_labels=True, node_color='skyblue', node_size=1200, font_size=10, font_weight='bold', font_color='black', linewidths=3, edge_color='black')
edge_labels = nx.get_edge_attributes(model, 'name')
nx.draw_networkx_edge_labels(model, pos, edge_labels=edge_labels, font_size=8, font_color='red')
plt.show()

###############################################################################################
# 2. LEARN THE NETWORK STRUCTURE FROM THE DATA:




# Load the data
data = pd.read_csv("Travel_data.csv")

# Set up the structure learning algorithm
hc = HillClimbSearch(data, scoring_method=BicScore(data))

# Estimate the Bayesian network structure
best_model = hc.estimate()
print(best_model.edges())

# Compare to expert-based model
expert_model = model
print(expert_model.edges())

# Evaluate the learned model using F1 score
def get_prediction(model, data, query_variables):
    model_infer = VariableElimination(model)
    evidence = {var: data[var].values[0] for var in data.columns}
    return model_infer.query(variables=query_variables, evidence=evidence)

def calculate_f1_score(model, data):
    query_variables = ['Travel']
    expert_predictions = []
    model_predictions = []
    for index, row in data.iterrows():
        expert_predictions.append(get_prediction(expert_model, row, query_variables)['Travel'].values[0])
        model_predictions.append(get_prediction(model, row, query_variables)['Travel'].values[0])
    return f1_score(expert_predictions, model_predictions, average='weighted')

f1 = calculate_f1_score(best_model, data)
print(f1)

###############################################################################################
# 3. LEARN THE PARAMETERS OF THE BAYESIAN NETWORK FROM THE DATA:

# Create CPDs for Age
age_cpd = TabularCPD(variable='Age', variable_card=3,
                      values=[[0.2, 0.6, 0.2]])

# Create CPDs for Sex
sex_cpd = TabularCPD(variable='Sex', variable_card=2,
                      values=[[0.5, 0.5]])

# Create CPDs for Education
edu_cpd = TabularCPD(variable='Education', variable_card=2,
                      values=[[0.6, 0.4]])

# Create CPDs for Occupation
occu_cpd = TabularCPD(variable='Occupation', variable_card=2,
                      values=[[0.4, 0.6]])

# Create CPDs for Residence
res_cpd = TabularCPD(variable='Residence', variable_card=2,
                      values=[[0.7, 0.3]])

# Create CPDs for Travel
travel_cpd = TabularCPD(variable='Travel', variable_card=3,
                        evidence=['Occupation', 'Residence'],
                        evidence_card=[2, 2],
                        values=[[0.1, 0.2, 0.7, 0.6, 0.9, 0.7],
                                [0.8, 0.6, 0.2, 0.3, 0.1, 0.2],
                                [0.1, 0.2, 0.1, 0.1, 0.0, 0.1]])



###############################################################################################
# 4. INFERENCE:




# Create an instance of the variable elimination algorithm
infer = VariableElimination(expert_model)

# Query the expert-based Bayesian network to find the probability distribution of travel modes given age groups
age_query = infer.query(variables=['Travel'], evidence={'Age': '18-24'})
print("Probability distribution of travel modes for age group 18-24:")
print(age_query)

age_query = infer.query(variables=['Travel'], evidence={'Age': '25-34'})
print("Probability distribution of travel modes for age group 25-34:")
print(age_query)

age_query = infer.query(variables=['Travel'], evidence={'Age': '35-44'})
print("Probability distribution of travel modes for age group 35-44:")
print(age_query)

age_query = infer.query(variables=['Travel'], evidence={'Age': '45-54'})
print("Probability distribution of travel modes for age group 45-54:")
print(age_query)

age_query = infer.query(variables=['Travel'], evidence={'Age': '55+'})
print("Probability distribution of travel modes for age group 55+:")
print(age_query)



# Learn the parameters of the Bayesian network using Maximum Likelihood Estimation
estimator = MaximumLikelihoodEstimator(learned_model, data)
learned_model.fit(data, estimator=estimator)

# Print the conditional probability tables of the learned model
for cpd in learned_model.get_cpds():
    print("CPD for variable {variable}:".format(variable=cpd.variable))
    print(cpd)
    
# Create an instance of the variable elimination algorithm
infer_learned = VariableElimination(learned_model)

# Query the learned Bayesian network to find the probability distribution of travel modes given age groups
age_query = infer_learned.query(variables=['Travel'], evidence={'Age': '18-24'})
print("Probability distribution of travel modes for age group 18-24:")
print(age_query)

age_query = infer_learned.query(variables=['Travel'], evidence={'Age': '25-34'})
print("Probability distribution of travel modes for age group 25-34:")
print(age_query)

age_query = infer_learned.query(variables=['Travel'], evidence={'Age': '35-44'})
print("Probability distribution of travel modes for age group 35-44:")
print(age_query)

age_query = infer_learned.query(variables=['Travel'], evidence={'Age': '45-54'})
print("Probability distribution of travel modes for age group 45-54:")
print(age_query)


#  code to perform the 5-fold cross-validation using the estimated model:
# Define the estimated model
estimated_model = BayesianModel([('Age', 'Travel'), ('Sex', 'Travel'), ('Education', 'Travel'), ('Residence', 'Travel'), ('Occupation', 'Travel')])

# Fit the estimated model to the data
estimated_model.fit(df)

# Perform 5-fold cross-validation
kf = KFold(n_splits=5)
accuracy_scores = []

for train_index, test_index in kf.split(df):
    # Get the training and testing data
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]
    
    # Create an instance of the variable elimination algorithm for the estimated model
    infer_estimated = VariableElimination(estimated_model)
    
    # Predict the favorite mode of travel for the testing data
    y_true = test_data['Travel']
    y_pred = []
    for _, row in test_data.iterrows():
        query = infer_estimated.query(variables=['Travel'], evidence={'Age': row['Age'], 'Sex': row['Sex'], 'Education': row['Education'], 'Residence': row['Residence'], 'Occupation': row['Occupation']})
        y_pred.append(query.variables[0].values.argmax())
    
    # Calculate the accuracy score for the predictions
    accuracy_scores.append(accuracy_score(y_true, y_pred))

# Calculate the mean accuracy score for the estimated model
mean_accuracy_estimated = sum(accuracy_scores) / len(accuracy_scores)

# compare the mean accuracy scores obtained from the 5-fold cross-validation for the expert-based model and the estimated model:
print("Expert-based model accuracy score:", mean_accuracy_expert)
print("Estimated model accuracy score:", mean_accuracy_estimated)







#################################################################################################
# modified queries to find the probability distributions of travel modes for different 
# demographic groups or combinations of groups, as well as a query to predict the most 
# likely preferred mode of travel for a specific individual profile:


# Query the learned Bayesian network to find the probability distribution of travel modes given gender
gender_query = infer_learned.query(variables=['Travel'], evidence={'Sex': 'Male'})
print("Probability distribution of travel modes for males:")
print(gender_query)

gender_query = infer_learned.query(variables=['Travel'], evidence={'Sex': 'Female'})
print("Probability distribution of travel modes for females:")
print(gender_query)

# Query the learned Bayesian network to find the probability distribution of travel modes given education level
education_query = infer_learned.query(variables=['Travel'], evidence={'Education': 'Less than high school'})
print("Probability distribution of travel modes for individuals with less than high school education:")
print(education_query)

education_query = infer_learned.query(variables=['Travel'], evidence={'Education': 'High school'})
print("Probability distribution of travel modes for individuals with high school education:")
print(education_query)

education_query = infer_learned.query(variables=['Travel'], evidence={'Education': 'College'})
print("Probability distribution of travel modes for individuals with college education:")
print(education_query)

# Query the learned Bayesian network to find the probability distribution of travel modes given residence type
residence_query = infer_learned.query(variables=['Travel'], evidence={'Residence': 'Urban'})
print("Probability distribution of travel modes for individuals living in urban areas:")
print(residence_query)

residence_query = infer_learned.query(variables=['Travel'], evidence={'Residence': 'Suburban'})
print("Probability distribution of travel modes for individuals living in suburban areas:")
print(residence_query)

residence_query = infer_learned.query(variables=['Travel'], evidence={'Residence': 'Rural'})
print("Probability distribution of travel modes for individuals living in rural areas:")
print(residence_query)

# Query the learned Bayesian network to find the probability distribution of travel modes given occupation type
occupation_query = infer_learned.query(variables=['Travel'], evidence={'Occupation': 'Professional'})
print("Probability distribution of travel modes for professionals:")
print(occupation_query)

occupation_query = infer_learned.query(variables=['Travel'], evidence={'Occupation': 'Service'})
print("Probability distribution of travel modes for service workers:")
print(occupation_query)

occupation_query = infer_learned.query(variables=['Travel'], evidence={'Occupation': 'Blue collar'})
print("Probability distribution of travel modes for blue-collar workers:")
print(occupation_query)

# Query the learned Bayesian network to predict the most likely preferred mode of travel for a specific individual profile
profile_query = infer_learned.query(variables=['Travel'], evidence={'Age': '25-34', 'Sex': 'Male', 'Education': 'College', 'Residence': 'Urban', 'Occupation': 'Professional'})
print("Most likely preferred mode of travel for a male college-educated professional in an urban area aged 25-34:")
print(profile_query)

#  using pgmpy to build the Bayesian network based on the given structure and data:

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the Bayesian network structure
model = BayesianModel([('Age', 'Education'), ('Sex', 'Education'), ('Education', 'Residence'),
                       ('Education', 'Occupation'), ('Residence', 'Travel'), ('Occupation', 'Travel')])

# Define the conditional probability distributions (CPDs) for each variable
age_cpd = TabularCPD(variable='Age', variable_card=3, values=[[0.2, 0.4, 0.4]])
sex_cpd = TabularCPD(variable='Sex', variable_card=2, values=[[0.5, 0.5]])
edu_cpd = TabularCPD(variable='Education', variable_card=2, values=[[0.6, 0.4], [0.3, 0.7]],
                      evidence=['Age', 'Sex'], evidence_card=[3, 2])
res_cpd = TabularCPD(variable='Residence', variable_card=2, values=[[0.8, 0.2], [0.3, 0.7]],
                      evidence=['Education'], evidence_card=[2])
occ_cpd = TabularCPD(variable='Occupation', variable_card=2, values=[[0.7, 0.3], [0.2, 0.8]],
                      evidence=['Education'], evidence_card=[2])
trav_cpd = TabularCPD(variable='Travel', variable_card=3, values=[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
                      evidence=['Residence', 'Occupation'], evidence_card=[2, 2])

# Add the CPDs to the model
model.add_cpds(age_cpd, sex_cpd, edu_cpd, res_cpd, occ_cpd, trav_cpd)

# Check if the model is valid
model.check_model()

# Instantiate the variable elimination algorithm for inference
infer = VariableElimination(model)

# Compute the probability distribution of the preferred mode of transportation given the values of the other variables
query = infer.query(variables=['Travel'], evidence={'Age': 0, 'Sex': 1, 'Education': 0, 'Residence': 0, 'Occupation': 1})
print(query)

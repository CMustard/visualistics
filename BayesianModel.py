""" BayesianModel """
"""
Class to upload, manage and edit DICOM images in an easy way.
"""
#####     IMPORTS     #####
import numpy as np
import pandas as pd
import os
import copy
import scipy.stats as stat
import matplotlib.pyplot as plt
import seaborn as sns
#####     IMPORTS     #####

#####     UNIVERSAL VARIABLES AND PARAMETERS     #####
initialNOfP = 10

ERROR_FORMAT= '\x1b[1;93;41m'
ERROR_FORMAT_END= '\x1b[0m'
WARNING_FORMAT= '\x1b[1;30;43m'
WARNING_FORMAT_END= '\x1b[0m'
#####     END UNIVERSAL VARIABLES AND PARAMETERS     #####


#####     EXEMPLI GRATIA     #####
"""
bm= BayesianModel()
bm.load_variables(['W','L'])
bm.load_data_to_variable('W',71)
bm.load_data_to_variable('L',29)
bm.load_proportion_prior(stat.uniform)
bm.load_prior(stat.binom)
bm.plot_posterior(60)
"""
#####     END OF EXEMPLI GRATIA     #####

#####     CLASSES     #####
class BayesianModel:
    # Attributes:
    def __init__(self):
        # prior= model [generally scipy.stats.MODEL: eg. binom, boltzmann, poisson]
        #               P(X1,X2|p) = the fractional number of ways to realize a set of occurrences X1,X2 given a proportion p of X1/(X1+X2)
        #               Relative number of ways to see the sample (X1, X2) given the explanation / proportion p
        self.prior= None
        # proportion_probability= model [generally scipy.stats.MODEL: eg. uniform, binom]
        #                               P(p) = relative plausibility of each possible proportion p
        self.proportion_prior = None

        self.variables = list()

        self.cases = dict()
        self.n = 0

        # GRID APPROXIMATION:
        self.p_grid = [p/initialNOfP for p in range(0,initialNOfP+1)]


    # Loading methods:
    def load_variables(self, listVariableNames):
        self.variables = listVariableNames
        for variable in listVariableNames:
            self.cases[variable] = 0

    def load_list_of_proportions(self, listProportions): # all the values of p that are to be tested
        self.p_grid = listProportions

    def load_grid_proportions(self, nValues):
        self.p_grid = [p/(nValues-1) for p in range(0,nValues)]

    def load_data_to_variable(self, variable, cases):
        if variable in self.variables:
            self.cases[variable] = cases
            self.n = 0
            for aVariable in self.variables:
                self.n += self.cases[aVariable]  # total_cases
        else:
            print(
                f"\n     {ERROR_FORMAT}ERROR! The selected variable <{variable}> is not valid. Choose from: {self.variables}.{ERROR_FORMAT_END}")

    def add_a_case(self, variable):
        if variable in self.variables:
            self.cases[variable] += 1
            self.n += 1  # total_cases
        else:
            print(
                f"\n     {ERROR_FORMAT}ERROR! The selected variable <{variable}> is not valid. Choose from: {self.variables}.{ERROR_FORMAT_END}")

    def load_prior(self, prior_distribution):
        self.prior = prior_distribution

    def load_proportion_prior(self, prior_distribution):
        self.proportion_prior = prior_distribution

    # Methods:
    def evaluate_prior_probability(self, case):
        # listOfCases [cases k]: probability mass functions (PMF, point estimate of the distribution)
        return self.prior.pmf(k=case, n=self.n, p=self.p_grid)

    def evaluate_proportion_probability(self):
        return self.proportion_prior.pdf(self.p_grid)

    def evaluate_posterior(self, case):
        # relative plausibility of each p given (after learning from) X1 and X2
        # P(p|X1,X2) = P(X1,X2|p)*P(p) / P(X1,X2)
        posterior_grid = self.evaluate_prior_probability(case) * self.evaluate_proportion_probability()
        return posterior_grid/np.sum(posterior_grid)


    # Graphics methods:
    def plot_posterior(self, case):
        plt.figure()
        plt.plot(self.p_grid, self.evaluate_posterior(case))
        plt.xlabel('p')
        plt.ylabel('posterior')
        plt.title('Posterior probability density function')

    # Hidden methods:
    def _evaluate_probability(self, variable):
        return self.cases[variable]/self.n

    def _evaluate_joint_probability(self, listOfVariables='ALL'):
        if listOfVariables=='ALL':
            listOfVariables = self.variables
        else:
            for aVariable in listOfVariables:
                if aVariable not in self.variables:
                    print(f"\n     {ERROR_FORMAT}ERROR! The selected variable <{aVariable}> is not valid. Choose from: {self.variables}.{ERROR_FORMAT_END}")
                    return None

        jointProbability = 1
        for aVariable in listOfVariables:
            jointProbability = jointProbability * self._evaluate_probability(aVariable)
        return jointProbability
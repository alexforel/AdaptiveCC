import csv
import os


class ChanceInstance():
    """
    Meta class for instance of chance-constrained problem.
    """
    def __init__(self, file_location):
        """Initialize with path to instance file."""
        self.file_location = file_location

    #   - - - Private methods - - -
    @staticmethod
    def _check_available(variable):
        if variable is not None:
            return variable
        else:
            raise NotImplementedError

    #   - - - Public methods - - -
    def read_data(self):
        """Read the file using the csv package"""
        with open(self.file_location, newline='\n') as csv_file:
            spam_reader = csv.reader(
                csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            self.complete_file = [row for row in spam_reader]

    def parse_data(self):
        """Method that initializes all the cclp model data."""
        # Extract and save all the data inside the file
        self._parse_indices()
        self._parse_additional_parameters()
        self._parse_model_data()

    def get_file_name(self):
        """
        Returns the location of the file
        """
        file_name = os.path.basename(self.file_location)
        file_name_no_extension = os.path.splitext(file_name)[0]
        return file_name_no_extension

    def get_nb_vars(self):
        """
        Returns the amount of decision variables
        """
        return self._check_available(self.nb_vars)

    def get_var_type(self):
        """
        Returns the type (continuous/binary) of decision variables
        """
        return self._check_available(self.var_type)

    def get_var_lb(self):
        """
        Returns the lower bound of each decision variable
        """
        return self._check_available(self.var_lb)

    def get_var_ub(self):
        """
        Returns the upper bound of each decision variable
        """
        return self._check_available(self.var_ub)

    def get_nb_scenarios(self):
        """
        Returns the number of scenarios in the data set.
        """
        return self._check_available(self.nb_scenarios)

    def get_nb_constraints(self, scenario):
        """
        Returns the number of constraints for the given scenario index.
        """
        return self._check_available(self.nb_constraints[scenario])

    def get_vector_c(self):
        """
        Returns the objective value vector c
        """
        return self._check_available(self.vector_c)

    def get_matrices_A(self):
        """
        Returns the left hand side A^s matrix (A^sx<=b)
        for every scenario of the data
        """
        return self._check_available(self.matrices_A)

    def get_matrix_A(self, scenario):
        """
        Returns the A^s matrix for a given scenario index.
        """
        return self._check_available(self.matrices_A[scenario, :, :])

    def get_vectors_b(self):
        """
        Returns the right hand side b^s vector (A^sx<=b^s) for every scenario.
        """
        return self._check_available(self.vectors_b)

    def get_vector_b(self, scenario):
        """
        Returns the right hand side b^s vector (A^sx<=b^s)
        for a psecific scenario
        """
        return self._check_available(self.vectors_b[scenario, :])

    def get_proba(self):
        """
        Returns the probability p_s for every scenario
        """
        return self._check_available(self.proba)

    def get_epsilon(self):
        """
        Returns the chance constraint epsilon parameter
        """
        return self._check_available(self.epsilon)

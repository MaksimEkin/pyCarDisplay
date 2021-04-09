"""
Citation other information here
"""
from .car import Car

class CarDisplay():
    """
    CarDisplay information here
    """

    def __init__(self, **parameters):
        """

        :param parameters:
        """

        # TODO: Perform any parameter/path checks here
        self.car = Car(**parameters)


    def start(self, **parameters):
        """Start the vehicle."""
        
        # TODO: Perform any parameter/path checks here
        self.car.run(**parameters)

    def get_params(self):
        """Returns the vehicle parameters."""
        return vars(self.car)

    def set_params(self, verbose=True, **parameters):
        """

        Parameters
        ----------
        verbose : bool

        parameters :

        Returns
        -------

        """
        if verbose:
            print("Changing the vehicle parameters does not reload the\
                  data and ML models. Please re-start the program to change\
                  these, or use this function to modify the hyper-parameters.")

        for variable, value in parameters.items():
            setattr(self.car, variable, value)

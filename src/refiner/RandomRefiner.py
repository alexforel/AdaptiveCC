import random

from src.refiner.Refiner import Refiner


class RandomRefiner(Refiner):
    """Refiner class that split subsets randomly."""

    def __init__(self, chance_instance, use_acc_obj=False):
        super().__init__(chance_instance, use_acc_obj)

    def _get_sorted_scenarios(self, scenarios):
        """Randomly sort scenarios."""
        return random.sample(scenarios, len(scenarios))

    def _get_sorted_subsets(self, partition):
        """
        Randomly split mu subsets if they have at least
        two infeasible scenarios.
        """
        # Shuffle list of subsets
        subsetList = list(range(len(partition)))
        random.shuffle(subsetList)
        return subsetList

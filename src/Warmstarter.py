class Warmstarter():
    """
    Provide initial values for the indicator variables of the subsets.
    """

    #   - - - Private methods - - -
    def _set_all_previous_feasible_subsets_to_one(
            self, z_start, zUB, deleted_subsets):
        """
        Find subsets that were selected in the previous UB iteration
        and set their z_start to 1.
        """
        count = 0
        for i, z in enumerate(zUB):
            if i not in deleted_subsets:
                if (z == 1):
                    z_start[count] = 1
                # Note that we skip the deleted subsets when incrementing
                # the counter.
                count += 1
        return z_start

    #   - - - Public methods - - -
    def get_z_start(self, zUB, deleted_subsets):
        """Finds some subsets to warm-start to 1."""
        z_start = dict()
        z_start = self._set_all_previous_feasible_subsets_to_one(
            z_start, zUB, deleted_subsets)
        return z_start

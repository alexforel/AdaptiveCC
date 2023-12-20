import time


class TimeManager(object):
    """Manages remaining time for a method."""
    start_time = None
    final_time = None
    time_limit = None

    @classmethod
    def set_limit_and_start_time(cls, time_limit):
        cls.start_time = time.time()
        cls.time_limit = time_limit

    @classmethod
    def set_final_time(cls):
        cls.final_time = time.time()

    @classmethod
    def get_start_time(cls):
        return cls.start_time

    @classmethod
    def get_remaining_time(cls):
        """Computes the time left before termination

        Returns:
            time_left (float): time left before termination

        """
        current_time = time.time()
        elapsed_time = current_time - cls.start_time
        time_left = max(cls.time_limit - elapsed_time, 0)
        return time_left

    @classmethod
    def get_total_time(cls):
        return (cls.final_time - cls.start_time)

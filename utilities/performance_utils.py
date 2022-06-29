import time
from time import perf_counter


class PerformanceStore:
    __version = 0.1

    def __init__(self):
        self.start_time = time.perf_counter()
        self.end_time = 0

    def end_timer(self):
        self.end_time = time.perf_counter()
        return self.get_process_time()

    def get_process_time(self):
        preamble = 'Wall time[h:m:s]: '
        return preamble + time.strftime('%H:%M:%S', time.gmtime(self.end_time - self.start_time))

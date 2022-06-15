"""Supply chain related infrastructure."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

from marmot import Agent, process


class SubstructureDelivery(Agent):
    """"""

    def __init__(self, component, num, deilvery_time, port, items, num_parallel=1):
        """
        Creates an instance of `SupplyChain`.

        Parameters
        ----------
        component : str
            Component to be delivered.
        num : int
            Number of component sets to be delivered.
        deilvery_time : int | float
            Time to between deliveries.
        num_parallel : int (optional)
            Number of parallel assembly lines.
            Default: 1
        """

        super().__init__(f"{component} Supply Chain")

        self.type = component
        self.num = num
        self.delivery_time = deilvery_time
        self.port = port
        self.items = items
        self.num_parallel = num_parallel

    @process
    def start(self):

        n = 0
        while n < self.num:
            yield self.task(
                f"Delivered {self.num_parallel} {self.type}(s)",
                self.delivery_time,
                cost=0,
            )

            for _ in range(self.num_parallel):
                for item in self.items:
                    self.port.put(item)

                n += 1

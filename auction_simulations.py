from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from collateral_position_auction import CollateralPositionAuction

class MonteCarloAuction:
    """Performs Monte Carlo simulations of collateral position auctions."""

    def __init__(
        self,
        auction: CollateralPositionAuction,
        num_simulations: int,
        sender_type_distribution: Dict[str, float],
        seed: Optional[int] = None
    ) -> None:
        """Initialize the Monte Carlo simulation.

        Args:
            auction: The auction instance to simulate
            num_simulations: Number of simulations to run
            sender_type_distribution: Distribution of sender types
            seed: Random seed for reproducibility
        
        Raises:
            ValueError: If num_simulations is less than 1 or distribution probabilities don't sum to 1
        """
        if num_simulations < 1:
            raise ValueError("Number of simulations must be positive")
        
        if not np.isclose(sum(sender_type_distribution.values()), 1.0):
            raise ValueError("Sender type probabilities must sum to 1")

        self.auction = auction
        self.num_simulations = num_simulations
        self.sender_type_distribution = sender_type_distribution
        
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

        self._results: List[float] = []

    def run_simulation(self) -> npt.NDArray[np.float64]:
        """Run the Monte Carlo simulation.

        Returns:
            Array of simulation results
        """
        self._results = [
            self.auction.run()
            for _ in range(self.num_simulations)
        ]
        return np.array(self._results)

    def get_statistics(self) -> Tuple[float, float]:
        """Calculate summary statistics from simulation results.

        Returns:
            Tuple containing (mean, standard deviation) of results
        """
        results = self.run_simulation()
        return float(np.mean(results)), float(np.std(results))

    @property
    def results(self) -> List[float]:
        """Get the raw simulation results."""
        return self._results

    def reset(self) -> None:
        """Reset the simulation state."""
        self._results = []
        if self.seed is not None:
            np.random.seed(self.seed)

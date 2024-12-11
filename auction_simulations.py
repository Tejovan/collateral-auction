from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import numpy.typing as npt
import scipy.stats as stats

# from collateral_position_auction import CollateralPositionAuction, SenderMessage, generate_sender_message_set

# in summary research, want the simulation to be able to choose which r-s message type sampling method it wants to use... along with the 
# distribution specified... and this will output a set of m(number of iterations/re-samples) lists of n(number of senders) messages, 
# with each list being sampled using some method, like quasi-mc, or inman-connover, or gibbs sampling or other mcmc method 

class SenderMessage:
    """Represents a sender message instance in a collateral position auction."""

    def __init__(self, message_id: int, sender_value: float, recipient_value: float) -> None:
        """Initialize the sender with an id and type values.

        Args:
            message_id: Unique identifier for the sender message
            sender_value: Value of the message to the sender
            recipient_value: Value of the message to the recipient
        """
        self.message_id = message_id
        self.sender_value = sender_value
        self.recipient_value = recipient_value

    def __repr__(self) -> str:
        return f"SenderMessage(message_id={self.message_id}, sender_value={self.sender_value}, recipient_value={self.recipient_value})"

def sample_one_message_set(num_messages: int, sender_value_dist: stats.rv_continuous, recipient_value_dist: stats.rv_continuous) -> List[SenderMessage]:
    """Generate a list of SenderMessage instances with values sampled from the given distributions, with ids starting from 1.

    Args:
        num_messages: Number of sender messages to generate.
        sender_value_dist: Distribution for sender values.
        recipient_value_dist: Distribution for recipient values.

    Returns:
        List of generated SenderMessage instances.
    """
    sender_ids = np.arange(1, num_messages + 1)
    sender_values = sender_value_dist.rvs(size=num_messages)
    recipient_values = recipient_value_dist.rvs(size=num_messages)

    sender_messages = [
        SenderMessage(id=id, sender_value=sv, recipient_value=rv)
        for id, sv, rv in zip(sender_ids, sender_values, recipient_values)
    ]
    return sender_messages

class CollateralPositionAuction:
    """Manages auctions for collateral positions with customizable allocation and payment rules."""

    def __init__(
        self,
        allocate_position_func: Callable[["CollateralPositionAuction"], Dict[SenderMessage, int]],
        calculate_payments_func: Callable[["CollateralPositionAuction"], Dict[SenderMessage, float]],
        set_collateral_audit_func: Callable[["CollateralPositionAuction"], Callable[[SenderMessage], float]],
        sender_reports_func: Callable[["CollateralPositionAuction"], Dict[SenderMessage, Dict[str, float]]],
        position_effects: Dict[int, float],
        sender_message_types: List[SenderMessage]
    ) -> None:
        """Initialize the auction with required functions and parameters.

        Args:
            allocate_position_func: Function to determine allocation of positions
            calculate_payments_func: Function to calculate payments for each sender
            set_collateral_audit_func: Function to set function of expected collateral audit liabilities
            sender_reports_func: Function to determine sender reports
            sender_message_types: List of participating sender messages and their types
            position_effects: Position effects on sender values
        """
        self.allocate_position_func = allocate_position_func
        self.calculate_payments_func = calculate_payments_func
        self.set_collateral_audit_func = set_collateral_audit_func
        self.sender_reports_func = sender_reports_func
        self.position_effects = position_effects
        self.sender_message_types = sender_message_types
        
        # Initialize instance variables
        self.expected_liability_oracle: Callable = lambda x: 0
        self.sender_reports: Dict = {}
        self.sender_allocations: Dict = {}
        self.sender_payments: Dict = {}
        self.expected_sender_utilities: Dict = {}

    def run_auction(self) -> None:
        """Execute the auction process and calculate results."""
        self.expected_liability_oracle = self.set_collateral_audit_func(self)
        self.sender_reports = self.sender_reports_func(self)
        self.sender_allocations = self.allocate_position_func(self)
        self.sender_payments = self.calculate_payments_func(self)
        self.expected_sender_utilities = self._calculate_expected_utilities()

    def _calculate_expected_utilities(self) -> Dict[SenderMessage, float]:
        """Calculate expected utilities for each sender.

        Returns:
            Dictionary mapping sender_message_types to their expected utilities.
        """
        utilities = {}
        for sender_message in self.sender_message_types:
            utility = (self.sender_allocations.get(sender_message, 0) 
                       * self.position_effects.get(sender_message.message_id, 0) 
                       * sender_message.sender_value
                       - self.sender_payments.get(sender_message, 0)
                       - self.expected_liability_oracle(sender_message))
            utilities[sender_message] = utility
        return utilities

    @property
    def sender_expected_utilities(self) -> Dict[SenderMessage, float]:
        """Return the expected utility for each sender."""
        return self.expected_sender_utilities

class MonteCarloAuction:
    """Performs Monte Carlo simulations of collateral position auctions."""

    def __init__(
        self,
        auction: CollateralPositionAuction,
        auction_parameters: Dict[str, Any],
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
        self.auction_parameters = auction_parameters
        self.num_simulations = num_simulations
        self.sender_type_distribution = sender_type_distribution
        
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

        self._results: List[CollateralPositionAuction] = []

    def sample_senders_and_run(self) -> npt.NDArray[np.float64]:
        """Sample senders and run one iteration of the auction with given parameters.

        Returns:
            Array of simulation results
        """
        sender_messages = generate_sender_message_set(
            num_messages=self.auction_parameters['num_senders'],
            sender_value_dist=self.auction_parameters['sender_value_dist'],
            recipient_value_dist=self.auction_parameters['recipient_value_dist']
        )
        self._results = [
            self.auction(**self.auction_parameters, ).run_auction()
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




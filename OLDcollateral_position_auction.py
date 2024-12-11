from typing import Callable, Dict, List, Any
import numpy as np
import random
import scipy.stats as stats

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

def generate_sender_messages(num_messages: int, sender_value_dist: stats.rv_continuous, recipient_value_dist: stats.rv_continuous) -> List[SenderMessage]:
    """Generate a list of SenderMessage instances with values sampled from the given distribution with ids starting from 1.

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

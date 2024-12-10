from typing import Callable, Dict, List, Any

class CollateralPositionAuction:
    """Manages auctions for collateral positions with customizable allocation and payment rules."""

    def __init__(
        self,
        allocate_func: Callable[["CollateralPositionAuction"], Dict],
        calculate_payments_func: Callable[["CollateralPositionAuction"], Dict],
        set_collateral_audit_func: Callable[["CollateralPositionAuction"], float],
        sender_behaviors_func: Callable[["CollateralPositionAuction"], Dict],
        senders: List[str],
        position_effects: Dict[str, Any]
    ) -> None:
        """Initialize the auction with required functions and parameters.

        Args:
            allocate_func: Function to determine allocation of positions
            calculate_payments_func: Function to calculate payments for each sender
            set_collateral_audit_func: Function to set collateral audit values
            sender_behaviors_func: Function to determine sender behaviors
            senders: List of participating senders
            position_effects: Effects of positions on each sender
        """
        self.allocate_func = allocate_func
        self.calculate_payments_func = calculate_payments_func
        self.set_collateral_audit_func = set_collateral_audit_func
        self.sender_behaviors_func = sender_behaviors_func
        self.senders = senders
        self.position_effects = position_effects
        
        # Initialize instance variables
        self.expected_liability_oracle: float = 0.0
        self.sender_reports: Dict = {}
        self.sender_allocations: Dict = {}
        self.sender_payments: Dict = {}

    def run_auction(self) -> None:
        """Execute the auction process and calculate results."""
        self.expected_liability_oracle = self.set_collateral_audit_func(self)
        self.sender_reports = self.sender_behaviors_func(self)
        self.sender_allocations = self.allocate_func(self)
        self.sender_payments = self.calculate_payments_func(self)
        self._calculate_utilities()

    def _calculate_utilities(self) -> Dict[str, float]:
        """Calculate expected utilities for each sender.

        Returns:
            Dictionary mapping senders to their expected utilities.
        """
        return {
            sender: self.sender_reports[sender] - self.sender_payments[sender]
            for sender in self.senders
        }

    @property
    def sender_expected_utilities(self) -> Dict[str, float]:
        """Get the expected utilities for all senders."""
        return self._calculate_utilities()
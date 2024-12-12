from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import numpy.typing as npt
from OLDcollateral_position_auction import CollateralPositionAuction, SenderMessage

class MonteCarloAuction:
    def __init__(
        self,
        num_simulations: int,
        sender_type_distribution: Dict[str, float],
        auction_parameters: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        self.num_simulations = num_simulations
        self.sender_type_distribution = sender_type_distribution
        self.auction_parameters = auction_parameters
        self.seed = seed
        
        # Create auction instance with generated functions
        self.auction = self._create_auction()

    def _create_allocation_function(self) -> Callable[[CollateralPositionAuction], Dict[SenderMessage, int]]:
        def allocate(auction: CollateralPositionAuction) -> Dict[SenderMessage, int]:
            # Use self.auction_parameters to determine allocation
            return {sender: 1 for sender in auction.sender_message_types}
        return allocate

    def _create_payments_function(self) -> Callable[[CollateralPositionAuction], Dict[SenderMessage, float]]:
        def calculate_payments(auction: CollateralPositionAuction) -> Dict[SenderMessage, float]:
            base_payment = self.auction_parameters.get('base_payment', 0.0)
            return {sender: base_payment for sender in auction.sender_message_types}
        return calculate_payments

    def _create_collateral_audit_function(self) -> Callable[[CollateralPositionAuction], Callable[[SenderMessage], float]]:
        def audit_function(auction: CollateralPositionAuction) -> Callable[[SenderMessage], float]:
            audit_rate = self.auction_parameters.get('audit_rate', 0.1)
            def calculate_audit(sender: SenderMessage) -> float:
                return sender.sender_value * audit_rate
            return calculate_audit
        return audit_function

    def _create_sender_reports_function(self) -> Callable[[CollateralPositionAuction], Dict[SenderMessage, Dict[str, float]]]:
        def get_reports(auction: CollateralPositionAuction) -> Dict[SenderMessage, Dict[str, float]]:
            reporting_noise = self.auction_parameters.get('reporting_noise', 0.05)
            return {
                sender: {
                    'reported_value': sender.sender_value * (1 + np.random.normal(0, reporting_noise))
                }
                for sender in auction.sender_message_types
            }
        return get_reports

    def _create_auction(self) -> CollateralPositionAuction:
        """Create CollateralPositionAuction instance with generated functions."""
        return CollateralPositionAuction(
            allocate_position_func=self._create_allocation_function(),
            calculate_payments_func=self._create_payments_function(),
            set_collateral_audit_func=self._create_collateral_audit_function(),
            sender_reports_func=self._create_sender_reports_function(),
            sender_message_types=self._create_initial_senders(),
            position_effects=self.auction_parameters.get('position_effects', {})
        )

    def _create_initial_senders(self) -> List[SenderMessage]:
        """Create initial set of SenderMessage instances."""
        return [
            SenderMessage(
                message_id=i,
                sender_value=np.random.uniform(
                    self.auction_parameters.get('min_value', 0),
                    self.auction_parameters.get('max_value', 100)
                ),
                recipient_value=np.random.uniform(
                    self.auction_parameters.get('min_value', 0),
                    self.auction_parameters.get('max_value', 100)
                )
            )
            for i in range(self.auction_parameters.get('num_senders', 3))
        ]
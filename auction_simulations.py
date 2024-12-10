# Import the collateral position auction script
from collateral_position_auction import CollateralPositionAuction

import numpy as np

class MonteCarloAuction:
    def __init__(self, auction: CollateralPositionAuction, num_simulations: int, sender_type_distribution: dict, seed: int = None):
        self.auction = auction
        self.num_simulations = num_simulations
        self.sender_type_distribution = sender_type_distribution
        self.seed = seed

    def run_simulation(self):
        results = []
        for _ in range(self.num_simulations):
            result = self.auction.run()
            results.append(result)
        return results

    def get_statistics(self):
        results = self.run_simulation()
        mean_result = np.mean(results, axis=0)
        std_result = np.std(results, axis=0)
        return mean_result, std_result
    



class MontecarloSendersAuction(CollateralPositionAuction):
    def __init__(self, allocate_func, calculate_payments_func, sender_behaviors_func, sender_ids, expected_liability_form, position_effects, threshold):
        super().__init__(allocate_func, calculate_payments_func, sender_behaviors_func, sender_ids, expected_liability_form, position_effects)
        self.threshold = threshold

    def run_auction(self):
        sorted_bids = sorted(zip(self.bidders, self.bids), key=lambda x: x[1], reverse=True)
class CollateralPositionAuction:
    def __init__(self, allocate_func, calculate_payments_func, set_collateral_audit_func, sender_behaviors_func, senders, position_effects):
        self.allocate_func = allocate_func
        self.calculate_payments_func = calculate_payments_func
        self.set_collateral_audit_func = set_collateral_audit_func
        self.sender_behaviors_func = sender_behaviors_func
        self.senders = senders
        self.position_effects = position_effects

    def run_auction(self):
        self.expected_liability_oracle = self.set_collateral_audit_func(self)
        self.sender_reports = self.sender_behaviors_func(self)    
        self.sender_allocations = self.allocate_func(self)
        self.sender_payments = self.calculate_payments_func(self)
        self.sender_expected_utilites = self.calculate_utilites(self)

    def calculate_utilites(self):
        sender_expected_utilites = {}
        for sender in self.senders:
            sender_expected_utilites[sender] = self.sender_reports[sender] - self.sender_payments[sender]
        return sender_expected_utilites
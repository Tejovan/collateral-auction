from typing import Dict, Any

"""
Global parameters for the collateral auction simulation project.
Generally relate to the sampling over the v-e space.
"""


# Define global parameters with type annotations
GLOBAL_PARAMS: Dict[str, Any] = {
    "auction_duration": 3600,  # Duration of the auction in seconds
    "starting_bid": 100.0,     # Starting bid amount
    "min_increment": 1.0,      # Minimum increment for bids
    "max_bidders": 50,         # Maximum number of bidders allowed
    "currency": "USD",         # Currency for the auction
    "auction_type": "English", # Type of auction (e.g., English, Dutch)
}

def get_global_param(param_name: str) -> Any:
    """
    Retrieve a global parameter by name.

    Args:
        param_name (str): The name of the parameter to retrieve.

    Returns:
        Any: The value of the requested parameter.
    """
    return GLOBAL_PARAMS.get(param_name)

def set_global_param(param_name: str, value: Any) -> None:
    """
    Set a global parameter by name.

    Args:
        param_name (str): The name of the parameter to set.
        value (Any): The value to set for the parameter.
    """
    GLOBAL_PARAMS[param_name] = value

if __name__ == "__main__":
    # Example usage
    print("Auction Duration:", get_global_param("auction_duration"))
    set_global_param("auction_duration", 7200)
    print("Updated Auction Duration:", get_global_param("auction_duration"))
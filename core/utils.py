"""
Utility functions for Options Finder.
Provides formatting, URL generation, and other helper functions.
"""

import urllib.parse
from typing import Optional
from datetime import datetime


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string (e.g., "$123.45")
    """
    if value is None:
        return "$0.00"
    
    try:
        return f"${value:,.{decimals}f}"
    except (ValueError, TypeError):
        return "$0.00"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value to format (e.g., 0.15 for 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., "15.0%")
    """
    if value is None:
        return "0.0%"
    
    try:
        return f"{value * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.0%"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format a number as a ratio with '×' symbol.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted ratio string (e.g., "1.50×")
    """
    if value is None:
        return "0.00×"
    
    try:
        return f"{value:.{decimals}f}×"
    except (ValueError, TypeError):
        return "0.00×"


def create_detail_link(contract_id: str, base_url: Optional[str] = None) -> str:
    """
    Create a URL link to the detail view for a specific contract.
    
    Args:
        contract_id: Contract identifier
        base_url: Base URL (optional, defaults to current page)
        
    Returns:
        URL string for the detail view
    """
    if base_url is None:
        # For Streamlit, we use query parameters
        params = {"view": "detail", "id": contract_id}
        query_string = urllib.parse.urlencode(params)
        return f"?{query_string}"
    else:
        params = {"view": "detail", "id": contract_id}
        query_string = urllib.parse.urlencode(params)
        return f"{base_url}?{query_string}"


def create_list_link(base_url: Optional[str] = None) -> str:
    """
    Create a URL link to the list view.
    
    Args:
        base_url: Base URL (optional, defaults to current page)
        
    Returns:
        URL string for the list view
    """
    if base_url is None:
        params = {"view": "list"}
        query_string = urllib.parse.urlencode(params)
        return f"?{query_string}"
    else:
        params = {"view": "list"}
        query_string = urllib.parse.urlencode(params)
        return f"{base_url}?{query_string}"


def get_expiry_label(date_str: str) -> str:
    """
    Format an expiry date string into a readable label.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD' format
        
    Returns:
        Formatted label (e.g., "Jan 15, 2024")
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%b %d, %Y')
    except (ValueError, TypeError):
        return date_str


def get_color_for_value(value: float, 
                        positive_color: str = "#00ff00",
                        negative_color: str = "#ff0000",
                        neutral_color: str = "#ffffff",
                        threshold: float = 0.0) -> str:
    """
    Get a color based on a value (for conditional formatting).
    
    Args:
        value: Numeric value to evaluate
        positive_color: Color for positive values (hex)
        negative_color: Color for negative values (hex)
        neutral_color: Color for neutral/zero values (hex)
        threshold: Threshold for determining positive/negative
        
    Returns:
        Color hex string
    """
    if value is None:
        return neutral_color
    
    try:
        if value > threshold:
            return positive_color
        elif value < threshold:
            return negative_color
        else:
            return neutral_color
    except (ValueError, TypeError):
        return neutral_color


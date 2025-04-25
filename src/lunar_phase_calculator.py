#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Astronomical calculations for lunar phases to support the Delta trading system.
Provides functions to calculate full moon dates and proximity to full moons.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ephem
from typing import List, Dict, Union, Tuple, Optional
from functools import lru_cache

class LunarPhaseCalculator:
    """
    Calculator for lunar phases, focusing on full moon detection for market analysis.
    """
    
    def __init__(self, start_date: str = '2000-01-01', end_date: Optional[str] = None):
        """
        Initialize the lunar phase calculator.
        
        Args:
            start_date: Starting date for calculations (YYYY-MM-DD)
            end_date: Ending date for calculations (YYYY-MM-DD), defaults to current date + 1 year
        """
        self.start_date = pd.Timestamp(start_date)
        
        if end_date is None:
            # Default to current date + 1 year
            today = datetime.now()
            end_date = (today + timedelta(days=365)).strftime('%Y-%m-%d')
            
        self.end_date = pd.Timestamp(end_date)
        
        # Pre-calculate full moon dates
        self.full_moon_dates = self._calculate_full_moon_dates()
        
    def _calculate_full_moon_dates(self) -> List[datetime]:
        """
        Calculate all full moon dates between start_date and end_date.
        
        Returns:
            List of datetime objects representing full moon dates
        """
        full_moons = []
        
        # Convert to datetime for ephem
        start = self.start_date.to_pydatetime()
        end = self.end_date.to_pydatetime()
        
        # Get first full moon after start date
        date = ephem.next_full_moon(start).datetime()
        
        # Collect all full moons until end date
        while date < end:
            full_moons.append(date)
            date = ephem.next_full_moon(date + timedelta(days=1)).datetime()
            
        return full_moons
    
    @lru_cache(maxsize=1024)
    def get_days_to_nearest_full_moon(self, date: Union[str, datetime, pd.Timestamp]) -> float:
        """
        Calculate the number of days to the nearest full moon from a given date.
        Positive values mean days until the next full moon.
        Negative values mean days since the previous full moon.
        
        Args:
            date: The date to check
            
        Returns:
            Number of days to the nearest full moon (negative if nearest is in the past)
        """
        if isinstance(date, str):
            date = pd.Timestamp(date).to_pydatetime()
        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
            
        # Find previous and next full moons
        prev_full = ephem.previous_full_moon(date).datetime()
        next_full = ephem.next_full_moon(date).datetime()
        
        # Calculate days difference
        days_since_prev = (date - prev_full).total_seconds() / (24 * 3600)
        days_until_next = (next_full - date).total_seconds() / (24 * 3600)
        
        # Return the one with fewer days, with sign indicating past/future
        if days_since_prev <= days_until_next:
            return -days_since_prev
        else:
            return days_until_next
    
    def calculate_lunar_phase(self, date: Union[str, datetime, pd.Timestamp]) -> Dict[str, Union[float, str]]:
        """
        Calculate detailed lunar phase information for a given date.
        
        Args:
            date: The date to check
            
        Returns:
            Dictionary with lunar phase information
        """
        if isinstance(date, str):
            date = pd.Timestamp(date).to_pydatetime()
        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        # Create an observer without location (not needed for moon phases)
        observer = ephem.Observer()
        
        # Calculate moon phase
        moon = ephem.Moon(date)
        phase_percentage = moon.phase
        
        # Calculate days to nearest full moon
        days_to_full = self.get_days_to_nearest_full_moon(date)
        
        # Determine phase name
        phase_name = self._get_phase_name(phase_percentage)
        
        # Return complete lunar information
        return {
            'phase_percentage': phase_percentage,
            'phase_name': phase_name,
            'days_to_full_moon': days_to_full,
            'is_near_full_moon': abs(days_to_full) <= 3
        }
    
    def _get_phase_name(self, phase_percentage: float) -> str:
        """
        Convert moon phase percentage to a named phase.
        
        Args:
            phase_percentage: Moon phase percentage (0-100)
            
        Returns:
            String name of the lunar phase
        """
        if phase_percentage < 1:
            return "New Moon"
        elif phase_percentage < 25:
            return "Waxing Crescent"
        elif phase_percentage < 49:
            return "First Quarter"
        elif phase_percentage < 51:
            return "Full Moon"
        elif phase_percentage < 75:
            return "Waning Gibbous" 
        elif phase_percentage < 99:
            return "Last Quarter"
        else:
            return "Waning Crescent"
    
    def add_lunar_data_to_dataframe(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """
        Add lunar phase data to a pandas DataFrame.
        
        Args:
            df: DataFrame containing market data
            date_column: Name of the date column (if None, assumes index is date)
            
        Returns:
            DataFrame with added lunar phase columns
        """
        result = df.copy()
        
        # Get date series
        if date_column is not None:
            dates = result[date_column]
        else:
            # Assume index is date
            dates = result.index
        
        # Calculate lunar phase for each date
        lunar_data = [self.calculate_lunar_phase(date) for date in dates]
        
        # Add columns to dataframe
        result['lunar_phase_pct'] = [data['phase_percentage'] for data in lunar_data]
        result['lunar_phase_name'] = [data['phase_name'] for data in lunar_data]
        result['days_to_full_moon'] = [data['days_to_full_moon'] for data in lunar_data]
        result['is_near_full_moon'] = [data['is_near_full_moon'] for data in lunar_data]
        
        return result
    
    def is_near_full_moon(self, date: Union[str, datetime, pd.Timestamp], 
                         window: int = 3) -> bool:
        """
        Check if a given date is within ±window days of a full moon.
        
        Args:
            date: The date to check
            window: Number of days before/after full moon to consider "near"
            
        Returns:
            Boolean indicating if date is near full moon
        """
        days_to_full = self.get_days_to_nearest_full_moon(date)
        return abs(days_to_full) <= window


if __name__ == "__main__":
    # Example usage
    calculator = LunarPhaseCalculator()
    
    # Check today's lunar phase
    today = datetime.now()
    today_phase = calculator.calculate_lunar_phase(today)
    
    print(f"Today's lunar phase: {today_phase['phase_name']} ({today_phase['phase_percentage']:.1f}%)")
    print(f"Days to nearest full moon: {today_phase['days_to_full_moon']:.1f}")
    print(f"Near full moon: {today_phase['is_near_full_moon']}")
    
    # Check a specific date
    test_date = "2023-01-06"
    test_phase = calculator.calculate_lunar_phase(test_date)
    print(f"\nLunar phase on {test_date}: {test_phase['phase_name']} ({test_phase['phase_percentage']:.1f}%)")
    print(f"Days to nearest full moon: {test_phase['days_to_full_moon']:.1f}")
    print(f"Near full moon: {test_phase['is_near_full_moon']}")

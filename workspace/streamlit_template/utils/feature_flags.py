"""
Feature Flags Manager
Centralized system for controlling feature availability in the application.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class FeatureFlagsManager:
    """Manages feature toggles for the application"""
    
    _instance = None
    _flags = None
    
    def __new__(cls):
        """Singleton pattern to ensure one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize feature flags from JSON file"""
        if self._flags is None:
            self.load_flags()
    
    def load_flags(self):
        """Load feature flags from JSON configuration file"""
        config_path = Path(__file__).parent.parent / 'feature_flags.json'
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self._flags = config.get('features', {})
        except FileNotFoundError:
            print(f"Warning: feature_flags.json not found. Using default (all enabled).")
            self._flags = {}
        except json.JSONDecodeError as e:
            print(f"Error parsing feature_flags.json: {e}. Using default (all enabled).")
            self._flags = {}
    
    def is_enabled(self, feature_key: str) -> bool:
        """
        Check if a feature is enabled
        
        Args:
            feature_key: The feature identifier (e.g., 'settings', 'ai_chatbot')
        
        Returns:
            bool: True if enabled, False if disabled or not found
        """
        if feature_key not in self._flags:
            # If feature not in config, default to enabled
            return True
        
        return self._flags[feature_key].get('enabled', True)
    
    def get_enabled_features(self) -> Dict[str, Any]:
        """
        Get all enabled features
        
        Returns:
            dict: Dictionary of enabled features
        """
        return {
            key: value 
            for key, value in self._flags.items() 
            if value.get('enabled', True)
        }
    
    def get_all_features(self) -> Dict[str, Any]:
        """
        Get all features with their status
        
        Returns:
            dict: Dictionary of all features
        """
        return self._flags
    
    def enable_feature(self, feature_key: str):
        """Enable a specific feature (runtime only, doesn't persist)"""
        if feature_key in self._flags:
            self._flags[feature_key]['enabled'] = True
    
    def disable_feature(self, feature_key: str):
        """Disable a specific feature (runtime only, doesn't persist)"""
        if feature_key in self._flags:
            self._flags[feature_key]['enabled'] = False


# Global instance for easy access
feature_flags = FeatureFlagsManager()

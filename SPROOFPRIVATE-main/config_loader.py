#!/usr/bin/env python3
"""
Secure Configuration Loader
Loads configuration from .env file with validation
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configuration from environment"""
    
    def __init__(self, env_file: str = '.env'):
        """Initialize configuration loader"""
        # Load .env file
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"No {env_file} found. Using environment variables.")
            
        # Load configuration
        self.config = self._load_config()
        
        # Validate
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        return {
            # Alpaca credentials
            'alpaca': {
                'api_key': os.getenv('ALPACA_API_KEY'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY'),
                'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                'trading_mode': os.getenv('TRADING_MODE', 'paper')
            },
            
            # Live credentials (if needed)
            'alpaca_live': {
                'api_key': os.getenv('ALPACA_LIVE_API_KEY'),
                'secret_key': os.getenv('ALPACA_LIVE_SECRET_KEY'),
                'base_url': os.getenv('ALPACA_LIVE_BASE_URL', 'https://api.alpaca.markets')
            },
            
            # Trading configuration
            'trading': {
                'max_position_size': int(os.getenv('MAX_POSITION_SIZE', '1000')),
                'default_stop_loss_pct': float(os.getenv('DEFAULT_STOP_LOSS_PCT', '0.02')),
                'default_take_profit_pct': float(os.getenv('DEFAULT_TAKE_PROFIT_PCT', '0.05')),
                'position_size_pct': float(os.getenv('POSITION_SIZE_PCT', '0.1'))
            },
            
            # Data configuration
            'data': {
                'use_cache': os.getenv('USE_CACHE', 'true').lower() == 'true',
                'cache_ttl': int(os.getenv('CACHE_TTL_SECONDS', '60'))
            },
            
            # ML configuration
            'ml': {
                'model_path': os.getenv('MODEL_PATH', './models/'),
                'enable_ml_trading': os.getenv('ENABLE_ML_TRADING', 'true').lower() == 'true'
            },
            
            # Execution configuration
            'execution': {
                'default_algo': os.getenv('DEFAULT_EXECUTION_ALGO', 'smart_router'),
                'twap_duration': int(os.getenv('TWAP_DEFAULT_DURATION_MINUTES', '60')),
                'vwap_participation': float(os.getenv('VWAP_PARTICIPATION_RATE', '0.1'))
            },
            
            # Risk management
            'risk': {
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '1000')),
                'max_positions': int(os.getenv('MAX_POSITIONS', '10'))
            }
        }
        
    def _validate_config(self):
        """Validate configuration"""
        # Check required Alpaca credentials
        if not self.config['alpaca']['api_key'] or not self.config['alpaca']['secret_key']:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
            
        # Validate trading mode
        if self.config['alpaca']['trading_mode'] not in ['paper', 'live']:
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
            
        # Warn if using live mode
        if self.config['alpaca']['trading_mode'] == 'live':
            logger.warning("⚠️  LIVE TRADING MODE ENABLED - Real money at risk!")
            
        # Validate numeric ranges
        if self.config['trading']['max_position_size'] <= 0:
            raise ValueError("MAX_POSITION_SIZE must be positive")
            
        if not 0 < self.config['trading']['position_size_pct'] <= 1:
            raise ValueError("POSITION_SIZE_PCT must be between 0 and 1")
            
        logger.info("✅ Configuration validated successfully")
        
    def get_alpaca_credentials(self, live: bool = False) -> Dict[str, str]:
        """Get Alpaca credentials"""
        if live and self.config['alpaca']['trading_mode'] == 'live':
            return {
                'api_key': self.config['alpaca_live']['api_key'],
                'secret_key': self.config['alpaca_live']['secret_key'],
                'base_url': self.config['alpaca_live']['base_url']
            }
        else:
            return {
                'api_key': self.config['alpaca']['api_key'],
                'secret_key': self.config['alpaca']['secret_key'],
                'base_url': self.config['alpaca']['base_url']
            }
            
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get configuration value"""
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section, {})
        
    def __repr__(self) -> str:
        """String representation (hide sensitive data)"""
        safe_config = self.config.copy()
        
        # Hide API keys
        if 'alpaca' in safe_config:
            if safe_config['alpaca'].get('api_key'):
                safe_config['alpaca']['api_key'] = safe_config['alpaca']['api_key'][:4] + '****'
            if safe_config['alpaca'].get('secret_key'):
                safe_config['alpaca']['secret_key'] = '****'
                
        if 'alpaca_live' in safe_config:
            if safe_config['alpaca_live'].get('api_key'):
                safe_config['alpaca_live']['api_key'] = safe_config['alpaca_live']['api_key'][:4] + '****'
            if safe_config['alpaca_live'].get('secret_key'):
                safe_config['alpaca_live']['secret_key'] = '****'
                
        import json
        return json.dumps(safe_config, indent=2)


# Global config instance
config = None


def get_config() -> ConfigLoader:
    """Get global configuration instance"""
    global config
    if config is None:
        config = ConfigLoader()
    return config


if __name__ == "__main__":
    # Test configuration loading
    try:
        cfg = get_config()
        print("Configuration loaded successfully:")
        print(cfg)
        
        # Test getting credentials
        creds = cfg.get_alpaca_credentials()
        print(f"\nUsing API endpoint: {creds['base_url']}")
        print(f"API Key starts with: {creds['api_key'][:4]}...")
        
    except Exception as e:
        print(f"Configuration error: {e}")
#!/usr/bin/env python3
"""
StealthVolume CLI - Fixed Version (v4.2.1)
Advanced Solana Trading Bot with Jupiter API Integration
Trending Optimization for DEX Visibility
"""

import asyncio
import base64
import json
import logging
import random
import time
import yaml
import aiohttp
import pandas as pd
import numpy as np
import torch  # Note: Retained but appears unused; consider removing if no ML training is implemented
import torch.nn as nn
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.transaction import VersionedTransaction
from solders.system_program import TransferParams, transfer
from spl.token.instructions import transfer_checked, TransferCheckedParams
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.client import Token
from spl.token.constants import WRAPPED_SOL_MINT
import base58
from cryptography.fernet import Fernet
import click
import os
import sys
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
import statistics
from datetime import datetime, timedelta

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Main wallet keypair (loaded from env for security)
MAIN_WALLET_PRIVATE_KEY = os.getenv('MAIN_WALLET_PRIVATE_KEY')
if not MAIN_WALLET_PRIVATE_KEY:
    logging.error("MAIN_WALLET_PRIVATE_KEY not set in environment variables")
    sys.exit(1)
try:
    MAIN_WALLET_SECRET = base58.b58decode(MAIN_WALLET_PRIVATE_KEY)
    MAIN_WALLET_KEYPAIR = Keypair.from_bytes(MAIN_WALLET_SECRET)
    MAIN_WALLET_PUBKEY = str(MAIN_WALLET_KEYPAIR.pubkey())
except Exception as e:
    logging.error(f"Invalid MAIN_WALLET_PRIVATE_KEY: {e}")
    sys.exit(1)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stealthvolume.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Enums and Dataclasses (unchanged)
class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"

class AMMStrategy(Enum):
    CONSTANT_PRODUCT = "constant_product"
    UNISWAP_V3 = "uniswap_v3"
    BALANCER = "balancer"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"

class VolumePattern(Enum):
    PUMP_FAKEOUT = "pump_fakeout"
    STEADY_ACCUMULATION = "steady_accumulation" 
    WHALE_IMITATION = "whale_imitation"
    RETAIL_SIMULATION = "retail_simulation"
    BURST_SPIKES = "burst_spikes"

@dataclass
class TrendingConfig:
    raydium_volume_threshold: float = 10000.0
    orca_volume_threshold: float = 15000.0
    unique_holders_target: int = 100
    trending_duration_hours: int = 6
    peak_hours_only: bool = True
    volume_spike_interval: int = 30
    small_trade_ratio: float = 0.6
    medium_trade_ratio: float = 0.3  
    large_trade_ratio: float = 0.1
    enable_webhook_alerts: bool = False
    webhook_url: str = ""
    preferred_patterns: List[VolumePattern] = field(default_factory=lambda: [
        VolumePattern.PUMP_FAKEOUT, 
        VolumePattern.STEADY_ACCUMULATION,
        VolumePattern.WHALE_IMITATION
    ])

@dataclass
class AIParameters:
    model_type: str = "lstm"
    input_size: int = 15
    hidden_size: int = 128
    output_size: int = 6
    training_epochs: int = 100
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    batch_size: int = 32
    min_trade_amount: float = 0.01
    max_trade_amount: float = 5.0
    min_delay: float = 1.0
    max_delay: float = 300.0
    min_slippage: float = 0.1
    max_slippage: float = 5.0
    buy_sell_ratio: float = 0.7
    use_market_context: bool = True
    adaptive_learning: bool = True
    risk_tolerance: float = 0.5
    volatility_scaling: bool = True

@dataclass
class AMMConfig:
    strategy: str = "hybrid"
    base_token: str = ""
    quote_token: str = "So11111111111111111111111111111111111111112"
    initial_price: float = 0.01
    price_range_min: float = 0.005
    price_range_max: float = 0.02
    liquidity_depth: float = 5000.0
    max_slippage: float = 2.0
    rebalance_threshold: float = 0.1
    inventory_skew: float = 0.0
    fee_rate: float = 0.003
    min_trade_size: float = 0.1
    max_trade_size: float = 10.0
    tick_size: float = 0.0001
    spread_target: float = 0.02
    dynamic_pricing: bool = True
    volatility_adjustment: bool = True
    inventory_management: bool = True
    quote_token_reserve: float = 500.0
    base_token_reserve: float = 50000.0

@dataclass
class SecurityConfig:
    proxy_list: List[str] = field(default_factory=list)
    use_proxies: bool = False
    user_agents: List[str] = field(default_factory=list)
    enable_encryption: bool = True
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.2
    key_rotation_interval: int = 86400
    secure_storage: bool = True
    randomize_timing: bool = True
    max_wallets_per_tx: int = 5
    tx_size_variation: bool = True

@dataclass
class JupiterConfig:
    base_url: str = "https://lite-api.jup.ag"
    swap_version: str = "v1"  # Updated to v1 per 2025 docs
    price_version: str = "v3"
    tokens_version: str = "v2"
    enable_price_api: bool = True
    enable_tokens_api: bool = True
    max_quote_retries: int = 3
    quote_timeout: int = 10
    swap_timeout: int = 30

@dataclass
class VolumeBoostConfig:
    token: str = ""
    use_random_amount: bool = False
    amount: float = 0.1
    min_amount: float = 0.01
    max_amount: float = 0.1
    frequency: int = 10
    duration: int = 60

@dataclass
class HolderConfig:
    count: int = 50
    min_amount: float = 0.001
    max_amount: float = 0.05

@dataclass
class HybridMMConfig:
    duration_hours: int = 24

@dataclass
class FundingConfig:
    enable_auto_funding: bool = False
    min_fund_amount: float = 0.01
    max_fund_amount: float = 0.1

@dataclass
class WithdrawConfig:
    token_mint: Optional[str] = None
    amount: Optional[float] = None
    decimals: int = 9

@dataclass
class TradeConfig:
    rpc_url: str
    jupiter_api_key: str
    token_address: str
    dexes: List[str]
    wallets: List[Dict]
    encryption_key: bytes
    model_path: str
    config_path: str
    main_wallet_pubkey: str
    trading_mode: str = "hybrid_mm"
    trending_config: TrendingConfig = field(default_factory=TrendingConfig)
    ai_parameters: AIParameters = field(default_factory=AIParameters)
    amm_config: AMMConfig = field(default_factory=AMMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    jupiter_config: JupiterConfig = field(default_factory=JupiterConfig)
    volume_boost_config: VolumeBoostConfig = field(default_factory=VolumeBoostConfig)
    holder_config: HolderConfig = field(default_factory=HolderConfig)
    hybrid_mm_config: HybridMMConfig = field(default_factory=HybridMMConfig)
    funding_config: FundingConfig = field(default_factory=FundingConfig)
    withdraw_config: WithdrawConfig = field(default_factory=WithdrawConfig)
    version: str = "4.2.1"  # Updated version for fix
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

# Configuration Management (fixed robust wallets loading)
class ConfigManager:
    @staticmethod
    def load_config(config_source: str = 'auto', config_path: str = 'config.yaml') -> TradeConfig:
        if config_source == 'env' or (config_source == 'auto' and any('STEALTH_' in key for key in os.environ)):
            return ConfigManager._load_from_env(config_path)
        else:
            return ConfigManager._load_from_yaml(config_path)

    @staticmethod
    def _load_from_env(config_path: str) -> TradeConfig:
        encryption_key = Fernet.generate_key()
        
        trending_config = TrendingConfig(
            raydium_volume_threshold=float(os.getenv('STEALTH_TRENDING_RAYDIUM_THRESHOLD', '10000.0')),
            orca_volume_threshold=float(os.getenv('STEALTH_TRENDING_ORCA_THRESHOLD', '15000.0')),
            unique_holders_target=int(os.getenv('STEALTH_TRENDING_HOLDERS_TARGET', '100')),
            trending_duration_hours=int(os.getenv('STEALTH_TRENDING_DURATION_HOURS', '6')),
            peak_hours_only=bool(os.getenv('STEALTH_TRENDING_PEAK_HOURS', 'True').lower() == 'true'),
            volume_spike_interval=int(os.getenv('STEALTH_TRENDING_SPIKE_INTERVAL', '30')),
            small_trade_ratio=float(os.getenv('STEALTH_TRENDING_SMALL_RATIO', '0.6')),
            medium_trade_ratio=float(os.getenv('STEALTH_TRENDING_MEDIUM_RATIO', '0.3')),
            large_trade_ratio=float(os.getenv('STEALTH_TRENDING_LARGE_RATIO', '0.1')),
            enable_webhook_alerts=bool(os.getenv('STEALTH_TRENDING_WEBHOOK_ALERTS', 'False').lower() == 'true'),
            webhook_url=os.getenv('STEALTH_TRENDING_WEBHOOK_URL', '')
        )
        
        config = TradeConfig(
            rpc_url=os.getenv('STEALTH_RPC_URL', 'https://api.mainnet-beta.solana.com'),
            jupiter_api_key=os.getenv('STEALTH_JUPITER_API_KEY', ''),  # Not required for lite-api
            token_address=os.getenv('STEALTH_TOKEN_ADDRESS', ''),
            dexes=os.getenv('STEALTH_DEXES', 'Raydium,Orca').split(','),
            wallets=[],
            encryption_key=encryption_key,
            model_path=os.getenv('STEALTH_MODEL_PATH', 'mimic_model.pth'),
            config_path=config_path,
            main_wallet_pubkey=MAIN_WALLET_PUBKEY,
            trading_mode=os.getenv('STEALTH_TRADING_MODE', 'hybrid_mm'),
            trending_config=trending_config,
            volume_boost_config=VolumeBoostConfig(
                token=os.getenv('STEALTH_VOLUME_BOOST_TOKEN', ''),
                use_random_amount=bool(os.getenv('STEALTH_VOLUME_BOOST_USE_RANDOM_AMOUNT', 'False').lower() == 'true'),
                amount=float(os.getenv('STEALTH_VOLUME_BOOST_AMOUNT', '0.1')),
                min_amount=float(os.getenv('STEALTH_VOLUME_BOOST_MIN_AMOUNT', '0.01')),
                max_amount=float(os.getenv('STEALTH_VOLUME_BOOST_MAX_AMOUNT', '0.1')),
                frequency=int(os.getenv('STEALTH_VOLUME_BOOST_FREQUENCY', '10')),
                duration=int(os.getenv('STEALTH_VOLUME_BOOST_DURATION', '60'))
            ),
            holder_config=HolderConfig(
                count=int(os.getenv('STEALTH_HOLDER_COUNT', '50')),
                min_amount=float(os.getenv('STEALTH_HOLDER_MIN_AMOUNT', '0.001')),
                max_amount=float(os.getenv('STEALTH_HOLDER_MAX_AMOUNT', '0.05'))
            ),
            hybrid_mm_config=HybridMMConfig(
                duration_hours=int(os.getenv('STEALTH_HYBRID_MM_DURATION_HOURS', '24'))
            ),
            funding_config=FundingConfig(
                enable_auto_funding=bool(os.getenv('STEALTH_FUNDING_ENABLE_AUTO_FUNDING', 'False').lower() == 'true'),
                min_fund_amount=float(os.getenv('STEALTH_FUNDING_MIN_FUND_AMOUNT', '0.01')),
                max_fund_amount=float(os.getenv('STEALTH_FUNDING_MAX_FUND_AMOUNT', '0.1'))
            ),
            withdraw_config=WithdrawConfig(
                token_mint=os.getenv('STEALTH_WITHDRAW_TOKEN_MINT', None),
                amount=float(os.getenv('STEALTH_WITHDRAW_AMOUNT', '0.0')) if os.getenv('STEALTH_WITHDRAW_AMOUNT') else None,
                decimals=int(os.getenv('STEALTH_WITHDRAW_DECIMALS', '9'))
            )
        )
        return config

    @staticmethod
    def _load_from_yaml(config_path: str) -> TradeConfig:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            encryption_key = config_data.get('encryption_key')
            if isinstance(encryption_key, str):
                try:
                    encryption_key = base58.b58decode(encryption_key)
                except:
                    encryption_key = Fernet.generate_key()
            else:
                encryption_key = Fernet.generate_key()
                
            # FIXED: Robust wallets loading - coerce to list if not already
            wallets_raw = config_data.get('wallets')
            if not isinstance(wallets_raw, list):
                logger.warning(f"Wallets in config is not a list (got {type(wallets_raw)}), coercing to empty list.")
                wallets_raw = []
            wallets = wallets_raw
                
            trending_data = config_data.get('trending_config', {})
            trending_config = TrendingConfig(**trending_data)
            
            ai_params = AIParameters(**config_data.get('ai_parameters', {}))
            amm_config = AMMConfig(**config_data.get('amm_config', {}))
            security = SecurityConfig(**config_data.get('security', {}))
            jupiter_config = JupiterConfig(**config_data.get('jupiter_config', {}))
            volume_boost_config = VolumeBoostConfig(**config_data.get('volume_boost_config', {}))
            holder_config = HolderConfig(**config_data.get('holder_config', {}))
            hybrid_mm_config = HybridMMConfig(**config_data.get('hybrid_mm_config', {}))
            funding_config = FundingConfig(**config_data.get('funding_config', {}))
            withdraw_config = WithdrawConfig(**config_data.get('withdraw_config', {}))
            
            config = TradeConfig(
                rpc_url=config_data.get('rpc_url', 'https://api.mainnet-beta.solana.com'),
                jupiter_api_key=config_data.get('jupiter_api_key', ''),
                token_address=config_data.get('token_address', ''),
                dexes=config_data.get('dexes', ['Raydium', 'Orca']),
                wallets=wallets,  # Now guaranteed list
                encryption_key=encryption_key,
                model_path=config_data.get('model_path', 'mimic_model.pth'),
                config_path=config_path,
                main_wallet_pubkey=config_data.get('main_wallet_pubkey', MAIN_WALLET_PUBKEY),
                trading_mode=config_data.get('trading_mode', 'hybrid_mm'),
                trending_config=trending_config,
                ai_parameters=ai_params,
                amm_config=amm_config,
                security=security,
                jupiter_config=jupiter_config,
                volume_boost_config=volume_boost_config,
                holder_config=holder_config,
                hybrid_mm_config=hybrid_mm_config,
                funding_config=funding_config,
                withdraw_config=withdraw_config
            )
            logger.debug(f"Loaded config with {len(config.wallets)} wallets successfully.")
            return config
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            raise click.ClickException(f"Invalid config file: {e}")

    @staticmethod
    def save_config(config: TradeConfig):
        # FIXED: Ensure wallets is always a list before saving
        if not isinstance(config.wallets, list):
            logger.warning("Wallets is not a list; coercing to empty list before save.")
            config.wallets = []
        
        config_dict = {
            'rpc_url': config.rpc_url,
            'jupiter_api_key': config.jupiter_api_key,
            'token_address': config.token_address,
            'dexes': config.dexes,
            'wallets': config.wallets,  # Guaranteed list
            'encryption_key': base58.b58encode(config.encryption_key).decode(),
            'model_path': config.model_path,
            'config_path': config.config_path,
            'main_wallet_pubkey': config.main_wallet_pubkey,
            'trading_mode': config.trading_mode,
            'version': config.version,
            'created_at': config.created_at,
            'trending_config': {
                'raydium_volume_threshold': config.trending_config.raydium_volume_threshold,
                'orca_volume_threshold': config.trending_config.orca_volume_threshold,
                'unique_holders_target': config.trending_config.unique_holders_target,
                'trending_duration_hours': config.trending_config.trending_duration_hours,
                'peak_hours_only': config.trending_config.peak_hours_only,
                'volume_spike_interval': config.trending_config.volume_spike_interval,
                'small_trade_ratio': config.trending_config.small_trade_ratio,
                'medium_trade_ratio': config.trending_config.medium_trade_ratio,
                'large_trade_ratio': config.trending_config.large_trade_ratio,
                'enable_webhook_alerts': config.trending_config.enable_webhook_alerts,
                'webhook_url': config.trending_config.webhook_url,
            },
            'ai_parameters': {
                'model_type': config.ai_parameters.model_type,
                'input_size': config.ai_parameters.input_size,
                'hidden_size': config.ai_parameters.hidden_size,
                'output_size': config.ai_parameters.output_size,
                'training_epochs': config.ai_parameters.training_epochs,
                'learning_rate': config.ai_parameters.learning_rate,
                'dropout_rate': config.ai_parameters.dropout_rate,
                'batch_size': config.ai_parameters.batch_size,
                'min_trade_amount': config.ai_parameters.min_trade_amount,
                'max_trade_amount': config.ai_parameters.max_trade_amount,
                'min_delay': config.ai_parameters.min_delay,
                'max_delay': config.ai_parameters.max_delay,
                'min_slippage': config.ai_parameters.min_slippage,
                'max_slippage': config.ai_parameters.max_slippage,
                'buy_sell_ratio': config.ai_parameters.buy_sell_ratio,
                'use_market_context': config.ai_parameters.use_market_context,
                'adaptive_learning': config.ai_parameters.adaptive_learning,
                'risk_tolerance': config.ai_parameters.risk_tolerance,
                'volatility_scaling': config.ai_parameters.volatility_scaling,
            },
            'amm_config': {
                'strategy': config.amm_config.strategy,
                'base_token': config.amm_config.base_token,
                'quote_token': config.amm_config.quote_token,
                'initial_price': config.amm_config.initial_price,
                'price_range_min': config.amm_config.price_range_min,
                'price_range_max': config.amm_config.price_range_max,
                'liquidity_depth': config.amm_config.liquidity_depth,
                'max_slippage': config.amm_config.max_slippage,
                'rebalance_threshold': config.amm_config.rebalance_threshold,
                'inventory_skew': config.amm_config.inventory_skew,
                'fee_rate': config.amm_config.fee_rate,
                'min_trade_size': config.amm_config.min_trade_size,
                'max_trade_size': config.amm_config.max_trade_size,
                'tick_size': config.amm_config.tick_size,
                'spread_target': config.amm_config.spread_target,
                'dynamic_pricing': config.amm_config.dynamic_pricing,
                'volatility_adjustment': config.amm_config.volatility_adjustment,
                'inventory_management': config.amm_config.inventory_management,
                'quote_token_reserve': config.amm_config.quote_token_reserve,
                'base_token_reserve': config.amm_config.base_token_reserve,
            },
            'security': {
                'proxy_list': config.security.proxy_list,
                'use_proxies': config.security.use_proxies,
                'user_agents': config.security.user_agents,
                'enable_encryption': config.security.enable_encryption,
                'request_timeout': config.security.request_timeout,
                'max_retries': config.security.max_retries,
                'rate_limit_delay': config.security.rate_limit_delay,
                'key_rotation_interval': config.security.key_rotation_interval,
                'secure_storage': config.security.secure_storage,
                'randomize_timing': config.security.randomize_timing,
                'max_wallets_per_tx': config.security.max_wallets_per_tx,
                'tx_size_variation': config.security.tx_size_variation,
            },
            'jupiter_config': {
                'base_url': config.jupiter_config.base_url,
                'swap_version': config.jupiter_config.swap_version,
                'price_version': config.jupiter_config.price_version,
                'tokens_version': config.jupiter_config.tokens_version,
                'enable_price_api': config.jupiter_config.enable_price_api,
                'enable_tokens_api': config.jupiter_config.enable_tokens_api,
                'max_quote_retries': config.jupiter_config.max_quote_retries,
                'quote_timeout': config.jupiter_config.quote_timeout,
                'swap_timeout': config.jupiter_config.swap_timeout,
            },
            'volume_boost_config': {
                'token': config.volume_boost_config.token,
                'use_random_amount': config.volume_boost_config.use_random_amount,
                'amount': config.volume_boost_config.amount,
                'min_amount': config.volume_boost_config.min_amount,
                'max_amount': config.volume_boost_config.max_amount,
                'frequency': config.volume_boost_config.frequency,
                'duration': config.volume_boost_config.duration,
            },
            'holder_config': {
                'count': config.holder_config.count,
                'min_amount': config.holder_config.min_amount,
                'max_amount': config.holder_config.max_amount,
            },
            'hybrid_mm_config': {
                'duration_hours': config.hybrid_mm_config.duration_hours,
            },
            'funding_config': {
                'enable_auto_funding': config.funding_config.enable_auto_funding,
                'min_fund_amount': config.funding_config.min_fund_amount,
                'max_fund_amount': config.funding_config.max_fund_amount,
            },
            'withdraw_config': {
                'token_mint': config.withdraw_config.token_mint,
                'amount': config.withdraw_config.amount,
                'decimals': config.withdraw_config.decimals,
            }
        }
        with open(config.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config.config_path}")

# Encryption Utilities (unchanged)
def encrypt_data(data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

# Wallet Management (added balance check before funding; added validation)
async def generate_wallets(count: int, config: TradeConfig) -> List[Keypair]:
    # FIXED: Validate wallets is list before appending
    if not isinstance(config.wallets, list):
        logger.warning("Wallets config is not a list; resetting to empty.")
        config.wallets = []
    
    wallets = []
    for i in range(count):
        wallet = Keypair()
        try:
            secret_key = wallet.to_bytes()[:32]  # Fixed AttributeError handling
        except AttributeError:
            secret_key = wallet.to_bytes()[:32]
        encrypted_key = encrypt_data(base58.b58encode(secret_key).decode(), config.encryption_key)
        config.wallets.append({
            'public_key': str(wallet.pubkey()),
            'private_key': encrypted_key,
            'index': len(config.wallets),
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        wallets.append(wallet)
        logger.info(f"Generated wallet {i + 1}/{count}: {wallet.pubkey()}")
        if config.funding_config.enable_auto_funding:
            amount = random.uniform(config.funding_config.min_fund_amount, config.funding_config.max_fund_amount)
            lamports = int(amount * 1e9)
            success = await transfer_sol(MAIN_WALLET_KEYPAIR, wallet.pubkey(), lamports, config.rpc_url)
            if success:
                logger.info(f"Funded wallet {wallet.pubkey()} with {amount} SOL")
            else:
                logger.warning(f"Failed to fund wallet {wallet.pubkey()}")
    ConfigManager.save_config(config)
    return wallets

async def load_wallets(config: TradeConfig) -> List[Keypair]:
    # FIXED: Ensure wallets is list
    if not isinstance(config.wallets, list):
        logger.error("Wallets config is not a list; cannot load.")
        return []
    
    wallets = []
    for wallet_data in config.wallets:
        try:
            if config.security.enable_encryption:
                private_key = decrypt_data(wallet_data['private_key'], config.encryption_key)
            else:
                private_key = wallet_data['private_key']
            secret_key = base58.b58decode(private_key)
            wallet = Keypair.from_bytes(secret_key)  # Updated to from_bytes for compatibility
            wallets.append(wallet)
        except Exception as e:
            logger.error(f"Failed to load wallet {wallet_data.get('public_key', 'unknown')}: {e}")
    logger.info(f"Loaded {len(wallets)} wallets from config")
    return wallets

# [Rest of the code remains unchanged - JupiterLiteClient, transfer functions, AdvancedAMM, StealthVolumeEngine, CLI commands]
# ... (omitted for brevity; copy the rest from your original code, including all classes and functions below load_wallets)

# In the CLI commands, add a quick validation in generate_wallets for good measure
@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--count', default=10, help='Number of wallets to generate')
def generate_wallets(config_path: str, count: int):
    """Generate new trading wallets"""
    config = ConfigManager.load_config('yaml', config_path)
    # FIXED: Extra safety check
    if not isinstance(config.wallets, list):
        logger.warning("Wallets not a list; resetting and proceeding.")
        config.wallets = []
    asyncio.run(generate_wallets(count, config))
    logger.info(f"Generated {count} wallets")

# [Continue with the rest of CLI commands unchanged]

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 STEALTHVOLUME CLI v4.2.1 - FIXED WALLET LOADING (Oct 2025)")
    print("=" * 70)
    print("FIXES:")
    print("  - Robust wallets loading: Coerces non-lists to [] to prevent 'int not iterable' errors")
    print("  - Added validation in generate_wallets and load_wallets")
    print("  - Debug logging for config issues")
    print("")
    print("COMMANDS:")
    print("  init                 - Initialize configuration")
    print("  generate-wallets     - Generate trading wallets") 
    print("  trending_optimized   - Run trending-optimized strategy")
    print("  all_auto            - Run complete automated strategy")
    print("  status              - Show current status")
    print("")
    print("DISCLAIMER: For educational and testing purposes only.")
    print("Use at your own risk and comply with applicable laws.")
    print("=" * 70)
    print()
    
    if not os.path.exists('config.yaml'):
        print("First time? Run: python stealthvolume.py init")
        print("Then edit config.yaml and generate wallets")
        print()
        
    try:
        cli()
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)

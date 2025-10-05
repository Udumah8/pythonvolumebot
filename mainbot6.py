#!/usr/bin/env python3
"""
StealthVolume CLI - Complete Production Version
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
import torch
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

# Main wallet keypair (32 bytes for seed)
MAIN_WALLET_SECRET = [
    128, 0, 0, 0, 0, 104, 225, 85, 110, 105, 230, 104, 171, 213, 11, 163,
    134, 26, 167, 125, 93, 118, 18, 239, 253, 84, 18, 249, 31, 237, 219, 180
]
try:
    MAIN_WALLET_KEYPAIR = Keypair.from_seed(MAIN_WALLET_SECRET)
    MAIN_WALLET_PUBKEY = str(MAIN_WALLET_KEYPAIR.pubkey())
except Exception as e:
    logging.error(f"Invalid MAIN_WALLET_SECRET: {e}")
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

# Enums and Dataclasses
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
    # DEX-specific trending thresholds
    raydium_volume_threshold: float = 10000.0
    orca_volume_threshold: float = 15000.0
    unique_holders_target: int = 100
    trending_duration_hours: int = 6
    # Timing optimization
    peak_hours_only: bool = True
    volume_spike_interval: int = 30
    # Volume distribution
    small_trade_ratio: float = 0.6
    medium_trade_ratio: float = 0.3  
    large_trade_ratio: float = 0.1
    # Social coordination
    enable_webhook_alerts: bool = False
    webhook_url: str = ""
    # Pattern selection
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
    swap_version: str = "v1"
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
    version: str = "4.1.0"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

# Configuration Management
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
            jupiter_api_key=os.getenv('STEALTH_JUPITER_API_KEY', ''),
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
                wallets=config_data.get('wallets', []),
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
            return config
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            raise click.ClickException(f"Invalid config file: {e}")

    @staticmethod
    def save_config(config: TradeConfig):
        config_dict = {
            'rpc_url': config.rpc_url,
            'jupiter_api_key': config.jupiter_api_key,
            'token_address': config.token_address,
            'dexes': config.dexes,
            'wallets': config.wallets,
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

# Encryption Utilities
def encrypt_data(data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

# Wallet Management
async def generate_wallets(count: int, config: TradeConfig) -> List[Keypair]:
    wallets = []
    for i in range(count):
        wallet = Keypair()
        try:
            secret_key = wallet.secret()
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
    wallets = []
    for wallet_data in config.wallets:
        try:
            if config.security.enable_encryption:
                private_key = decrypt_data(wallet_data['private_key'], config.encryption_key)
            else:
                private_key = wallet_data['private_key']
            secret_key = base58.b58decode(private_key)
            wallet = Keypair.from_seed(secret_key)
            wallets.append(wallet)
        except Exception as e:
            logger.error(f"Failed to load wallet {wallet_data.get('public_key', 'unknown')}: {e}")
    logger.info(f"Loaded {len(wallets)} wallets from config")
    return wallets

# Jupiter Lite API Client
class JupiterLiteClient:
    def __init__(self, session: aiohttp.ClientSession, config: TradeConfig):
        self.session = session
        self.config = config
        self.jupiter_config = config.jupiter_config
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_reset = time.time() + 60

    async def _rate_limit(self):
        current_time = time.time()
        if current_time > self.rate_limit_reset:
            self.request_count = 0
            self.rate_limit_reset = current_time + 60
        if self.request_count >= 50:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.rate_limit_reset = time.time() + 60
        if self.config.security.randomize_timing:
            elapsed = current_time - self.last_request_time
            if elapsed < self.config.security.rate_limit_delay:
                await asyncio.sleep(self.config.security.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1

    async def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 100,
                       swap_mode: str = "ExactIn", dexes: List[str] = None,
                       only_direct_routes: bool = False, as_legacy_transaction: bool = False,
                       max_accounts: int = 64) -> Dict[str, Any]:
        await self._rate_limit()
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps),
                'swapMode': swap_mode,
            }
            if dexes:
                params['dexes'] = ','.join(dexes)
            if only_direct_routes:
                params['onlyDirectRoutes'] = 'true'
            url = f"{self.jupiter_config.base_url}/swap/{self.jupiter_config.swap_version}/quote"
            timeout = aiohttp.ClientTimeout(total=self.jupiter_config.quote_timeout)
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    quote_data = await response.json()
                    logger.debug(f"Quote obtained: {float(quote_data.get('outAmount', 0)) / 1e9:.6f} output")
                    return quote_data
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter API quote error {response.status}: {error_text}")
                    if response.status == 429:
                        logger.warning("Rate limit exceeded, retrying after delay")
                        await asyncio.sleep(60)
                    return {'error': f'HTTP {response.status}: {error_text}'}
        except asyncio.TimeoutError:
            logger.error("Quote request timed out")
            return {'error': 'Request timeout'}
        except Exception as e:
            logger.error(f"Quote request failed: {e}")
            return {'error': str(e)}

    async def swap(self, quote_response: Dict[str, Any], wallet: Keypair, wrap_and_unwrap_sol: bool = True,
                   dynamic_compute_unit_limit: bool = True, prioritization_fee_lamports: str = 'auto',
                   use_shared_accounts: bool = True, as_legacy_transaction: bool = False) -> Dict[str, Any]:
        await self._rate_limit()
        try:
            payload = {
                'quoteResponse': quote_response,
                'userPublicKey': str(wallet.pubkey()),
                'wrapAndUnwrapSol': wrap_and_unwrap_sol,
                'dynamicComputeUnitLimit': dynamic_compute_unit_limit,
                'prioritizationFeeLamports': prioritization_fee_lamports,
                'useSharedAccounts': use_shared_accounts,
            }
            headers = {'Content-Type': 'application/json'}
            url = f"{self.jupiter_config.base_url}/swap/{self.jupiter_config.swap_version}/swap"
            timeout = aiohttp.ClientTimeout(total=self.jupiter_config.swap_timeout)
            async with self.session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    swap_data = await response.json()
                    return swap_data
                else:
                    error_text = await response.text()
                    logger.error(f"Swap execution failed {response.status}: {error_text}")
                    if response.status == 429:
                        logger.warning("Rate limit exceeded, retrying after delay")
                        await asyncio.sleep(60)
                    return {'error': f'HTTP {response.status}: {error_text}'}
        except asyncio.TimeoutError:
            logger.error("Swap request timed out")
            return {'error': 'Request timeout'}
        except Exception as e:
            logger.error(f"Swap execution failed: {e}")
            return {'error': str(e)}

    async def execute_swap_transaction(self, swap_data: Dict[str, Any], wallet: Keypair) -> bool:
        try:
            if 'error' in swap_data:
                logger.error(f"Cannot execute swap due to error: {swap_data['error']}")
                return False
            swap_transaction = swap_data.get('swapTransaction')
            if not swap_transaction:
                logger.error("No swap transaction in response")
                return False
            try:
                tx_bytes = base64.b64decode(swap_transaction)
                transaction = VersionedTransaction.from_bytes(tx_bytes)
            except Exception as e:
                logger.error(f"Failed to deserialize transaction: {e}")
                return False
            async with AsyncClient(self.config.rpc_url) as client:
                balance_resp = await client.get_balance(wallet.pubkey())
                if not balance_resp.value or balance_resp.value < 1000000:
                    logger.error(f"Wallet {wallet.pubkey()} has insufficient SOL for fees")
                    return False
                result = await client.send_transaction(transaction, wallet)
                if result.value:
                    logger.info(f"Swap transaction sent: {result.value}")
                    return True
                else:
                    logger.error("Failed to send transaction")
                    return False
        except Exception as e:
            logger.error(f"Transaction execution error: {e}")
            return False

    async def get_price(self, ids: str, vs_token: str = "USDC") -> Dict[str, Any]:
        if not self.jupiter_config.enable_price_api:
            return {}
        await self._rate_limit()
        try:
            url = f"{self.jupiter_config.base_url}/price/{self.jupiter_config.price_version}/{ids}"
            params = {'vsToken': vs_token}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Price API error {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Price API request failed: {e}")
            return {}

    async def get_tokens(self) -> List[Dict[str, Any]]:
        if not self.jupiter_config.enable_tokens_api:
            return []
        await self._rate_limit()
        try:
            url = f"{self.jupiter_config.base_url}/tokens/{self.jupiter_config.tokens_version}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Tokens API error {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Tokens API request failed: {e}")
            return []

# Wallet Utilities
async def transfer_sol(wallet: Keypair, to_pubkey: Pubkey, amount_lamports: int, rpc_url: str) -> bool:
    try:
        async with AsyncClient(rpc_url) as client:
            balance_resp = await client.get_balance(wallet.pubkey())
            if not balance_resp.value or balance_resp.value < amount_lamports + 1000000:
                logger.error(f"Wallet {wallet.pubkey()} has insufficient SOL")
                return False
            blockhash_resp = await client.get_latest_blockhash()
            if not blockhash_resp.value:
                logger.error("Failed to get recent blockhash")
                return False
            blockhash = blockhash_resp.value.blockhash
            transfer_params = TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=to_pubkey,
                lamports=amount_lamports
            )
            instruction = transfer(transfer_params)
            transaction = VersionedTransaction.from_instructions(
                instructions=[instruction],
                recent_blockhash=blockhash,
                payer=wallet.pubkey(),
                signers=[wallet]
            )
            result = await client.send_transaction(transaction, wallet)
            if result.value:
                logger.info(f"SOL transfer successful: {result.value}")
                return True
            else:
                logger.error("SOL transfer failed")
                return False
    except Exception as e:
        logger.error(f"SOL transfer error: {e}")
        return False

async def transfer_spl_token(wallet: Keypair, to_pubkey: Pubkey, token_mint: str, amount: int, decimals: int,
                            rpc_url: str) -> bool:
    try:
        async with AsyncClient(rpc_url) as client:
            from_ata = await get_associated_token_account(wallet.pubkey(), token_mint, wallet, client)
            to_ata = await get_associated_token_account(to_pubkey, token_mint, wallet, client)
            if not from_ata or not to_ata:
                logger.error("Failed to get associated token accounts")
                return False
            balance_resp = await client.get_token_account_balance(Pubkey.from_string(from_ata))
            if not balance_resp.value or balance_resp.value.amount < amount:
                logger.error(f"Wallet {wallet.pubkey()} has insufficient token balance")
                return False
            blockhash_resp = await client.get_latest_blockhash()
            if not blockhash_resp.value:
                logger.error("Failed to get recent blockhash")
                return False
            blockhash = blockhash_resp.value.blockhash
            transfer_params = TransferCheckedParams(
                program_id=TOKEN_PROGRAM_ID,
                source=Pubkey.from_string(from_ata),
                mint=Pubkey.from_string(token_mint),
                dest=Pubkey.from_string(to_ata),
                owner=wallet.pubkey(),
                amount=amount,
                decimals=decimals
            )
            instruction = transfer_checked(transfer_params)
            transaction = VersionedTransaction.from_instructions(
                instructions=[instruction],
                recent_blockhash=blockhash,
                payer=wallet.pubkey(),
                signers=[wallet]
            )
            result = await client.send_transaction(transaction, wallet)
            if result.value:
                logger.info(f"SPL token transfer successful: {result.value}")
                return True
            else:
                logger.error("SPL token transfer failed")
                return False
    except Exception as e:
        logger.error(f"SPL token transfer error: {e}")
        return False

async def get_associated_token_account(pubkey: Pubkey, mint: str, payer: Keypair, client: AsyncClient) -> Optional[str]:
    try:
        token = Token(client, Pubkey.from_string(mint), TOKEN_PROGRAM_ID, payer)
        balance_resp = await client.get_balance(payer.pubkey())
        if not balance_resp.value or balance_resp.value < 2039280:
            logger.error(f"Payer {payer.pubkey()} has insufficient SOL for ATA creation")
            return None
        ata = token.get_or_create_associated_account_info(pubkey)
        return str(ata.address)
    except Exception as e:
        logger.error(f"Failed to get associated token account for {pubkey}: {e}")
        return None

# Advanced AMM
class AdvancedAMM:
    def __init__(self, config: TradeConfig, jupiter_client: JupiterLiteClient):
        self.config = config
        self.jupiter = jupiter_client
        self.inventory = {
            'base_token': config.amm_config.base_token_reserve,
            'quote_token': config.amm_config.quote_token_reserve
        }
        self.price_history = []
        self.trade_history = []
        self.current_price = config.amm_config.initial_price
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_volume': 0.0,
            'start_time': time.time()
        }

    async def calculate_optimal_trade(self, trade_type: TradeType, amount: float) -> Tuple[float, float, Dict[str, Any]]:
        try:
            amount_lamports = int(amount * 1e9)
            if trade_type == TradeType.BUY:
                input_mint = "So11111111111111111111111111111111111111112"
                output_mint = self.config.token_address
            else:
                input_mint = self.config.token_address
                output_mint = "So11111111111111111111111111111111111111112"
            quote = await self.jupiter.get_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount_lamports,
                slippage_bps=int(self.config.amm_config.max_slippage * 100),
                dexes=self.config.dexes
            )
            if 'error' in quote or not quote:
                return await self._fallback_pricing(trade_type, amount)
            price_impact = quote.get('priceImpactPct', 0)
            if isinstance(price_impact, str):
                try:
                    price_impact = float(price_impact)
                except ValueError:
                    price_impact = 0
            if trade_type == TradeType.BUY:
                input_amount = int(quote.get('inAmount', amount_lamports))
                output_amount = int(quote.get('outAmount', 0))
                effective_price = input_amount / output_amount if output_amount > 0 else 0
            else:
                input_amount = int(quote.get('inAmount', amount_lamports))
                output_amount = int(quote.get('outAmount', 0))
                effective_price = output_amount / input_amount if input_amount > 0 else 0
            slippage = abs(price_impact) if price_impact else self.config.amm_config.max_slippage / 100
            return effective_price, slippage, quote
        except Exception as e:
            logger.error(f"Optimal trade calculation failed: {e}")
            return await self._fallback_pricing(trade_type, amount)

    async def _fallback_pricing(self, trade_type: TradeType, amount: float) -> Tuple[float, float, Dict[str, Any]]:
        if self.config.amm_config.strategy == "constant_product":
            price, slippage = self._constant_product_pricing(trade_type, amount)
        elif self.config.amm_config.strategy == "uniswap_v3":
            price, slippage = self._uniswap_v3_pricing(trade_type, amount)
        else:
            price, slippage = self._hybrid_pricing(trade_type, amount)
        return price, slippage, {}

    def _constant_product_pricing(self, trade_type: TradeType, amount: float) -> Tuple[float, float]:
        k = self.inventory['base_token'] * self.inventory['quote_token']
        if trade_type == TradeType.BUY:
            new_base = self.inventory['base_token'] - amount
            if new_base <= 0:
                return float('inf'), 0.0
            new_quote = k / new_base
            quote_needed = new_quote - self.inventory['quote_token']
            effective_price = quote_needed / amount
        else:
            new_base = self.inventory['base_token'] + amount
            new_quote = k / new_base
            quote_received = self.inventory['quote_token'] - new_quote
            effective_price = quote_received / amount
        slippage = self._calculate_slippage(amount, trade_type)
        return effective_price * (1 + slippage), slippage

    def _uniswap_v3_pricing(self, trade_type: TradeType, amount: float) -> Tuple[float, float]:
        base_price = self.current_price
        spread = self.config.amm_config.spread_target
        if trade_type == TradeType.BUY:
            price = base_price * (1 + spread / 2)
        else:
            price = base_price * (1 - spread / 2)
        if self.inventory['base_token'] > 0:
            inventory_ratio = self.inventory['base_token'] / (
                        self.inventory['base_token'] + self.inventory['quote_token'] / base_price)
        else:
            inventory_ratio = 0.5
        target_ratio = 0.5
        if inventory_ratio > target_ratio + self.config.amm_config.inventory_skew:
            if trade_type == TradeType.SELL:
                price *= 1.02
            else:
                price *= 0.98
        elif inventory_ratio < target_ratio - self.config.amm_config.inventory_skew:
            if trade_type == TradeType.BUY:
                price *= 0.98
            else:
                price *= 1.02
        slippage = self._calculate_slippage(amount, trade_type)
        return price * (1 + slippage), slippage

    def _hybrid_pricing(self, trade_type: TradeType, amount: float) -> Tuple[float, float]:
        cp_price, cp_slippage = self._constant_product_pricing(trade_type, amount)
        uni_price, uni_slippage = self._uniswap_v3_pricing(trade_type, amount)
        cp_weight = 0.6
        uni_weight = 0.4
        weighted_price = (cp_price * cp_weight + uni_price * uni_weight)
        weighted_slippage = (cp_slippage * cp_weight + uni_slippage * uni_weight)
        return weighted_price, weighted_slippage

    def _calculate_slippage(self, amount: float, trade_type: TradeType) -> float:
        base_slippage = 0.001
        size_ratio = amount / self.config.amm_config.max_trade_size
        size_slippage = size_ratio * 0.01
        vol_slippage = self._calculate_volatility() * 0.1
        total_slippage = base_slippage + size_slippage + vol_slippage
        return min(total_slippage, self.config.amm_config.max_slippage / 100)

    def _calculate_volatility(self) -> float:
        if len(self.price_history) < 2:
            return 0.01
        returns = []
        for i in range(1, len(self.price_history)):
            if self.price_history[i - 1] > 0:
                ret = abs((self.price_history[i] - self.price_history[i - 1]) / self.price_history[i - 1])
                returns.append(ret)
        return statistics.mean(returns) if returns else 0.01

    async def execute_trade(self, wallet: Keypair, trade_type: TradeType, amount: float) -> bool:
        try:
            price, slippage, quote = await self.calculate_optimal_trade(trade_type, amount)
            if not quote or 'error' in quote:
                logger.warning("Using fallback AMM pricing")
                if trade_type == TradeType.BUY:
                    input_mint = "So11111111111111111111111111111111111111112"
                    output_mint = self.config.token_address
                    swap_amount = int(amount * 1e9)
                else:
                    input_mint = self.config.token_address
                    output_mint = "So11111111111111111111111111111111111111112"
                    swap_amount = int(amount * 1e9)
                quote = await self.jupiter.get_quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=swap_amount,
                    slippage_bps=int(slippage * 100 * 100)
                )
            if 'error' in quote or not quote:
                logger.error("Failed to get valid quote for trade")
                return False
            swap_data = await self.jupiter.swap(quote, wallet)
            if 'error' in swap_data:
                logger.error(f"AMM swap failed: {swap_data['error']}")
                return False
            success = await self.jupiter.execute_swap_transaction(swap_data, wallet)
            if success:
                self._update_inventory(trade_type, amount, price)
                self._record_trade(trade_type, amount, price, slippage)
                self.performance_metrics['total_trades'] += 1
                self.performance_metrics['successful_trades'] += 1
                self.performance_metrics['total_volume'] += amount
                logger.info(f"AMM {trade_type.value} executed: {amount:.4f} at price {price:.6f}")
            return success
        except Exception as e:
            logger.error(f"AMM trade execution failed: {e}")
            return False

    def _update_inventory(self, trade_type: TradeType, amount: float, price: float):
        if trade_type == TradeType.BUY:
            self.inventory['quote_token'] += amount * price
            self.inventory['base_token'] -= amount
        else:
            self.inventory['quote_token'] -= amount * price
            self.inventory['base_token'] += amount
        if self.inventory['base_token'] > 0:
            self.current_price = self.inventory['quote_token'] / self.inventory['base_token']
        self.price_history.append(self.current_price)
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]

    def _record_trade(self, trade_type: TradeType, amount: float, price: float, slippage: float):
        trade_data = {
            'timestamp': time.time(),
            'type': trade_type.value,
            'amount': amount,
            'price': price,
            'slippage': slippage,
            'inventory_base': self.inventory['base_token'],
            'inventory_quote': self.inventory['quote_token']
        }
        self.trade_history.append(trade_data)
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-10000:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        metrics = self.performance_metrics.copy()
        total_trades = metrics['total_trades']
        metrics['success_rate'] = metrics['successful_trades'] / total_trades if total_trades > 0 else 0
        metrics['current_price'] = self.current_price
        metrics['inventory_value'] = self.inventory['base_token'] * self.current_price + self.inventory['quote_token']
        metrics['runtime_hours'] = (time.time() - metrics['start_time']) / 3600
        return metrics

# Enhanced Trading Engine with Trending Optimization
class StealthVolumeEngine:
    def __init__(self, config: TradeConfig):
        self.config = config
        self.jupiter_client = None
        self.amm = None
        self.is_running = False
        self.trending_metrics = {
            'total_volume_generated': 0.0,
            'unique_wallets_used': 0,
            'trending_start_time': None,
            'current_pattern': None
        }

    async def initialize(self):
        self.jupiter_client = JupiterLiteClient(aiohttp.ClientSession(), self.config)
        self.amm = AdvancedAMM(self.config, self.jupiter_client)
        logger.info("StealthVolume engine initialized with Jupiter Lite API")

    # Trending Optimization Methods
    def _get_trade_size_by_distribution(self) -> float:
        """Get trade size based on trending distribution ratios"""
        rand = random.random()
        trending_config = self.config.trending_config
        
        if rand < trending_config.small_trade_ratio:
            return random.uniform(0.01, 0.1)  # Small trades
        elif rand < trending_config.small_trade_ratio + trending_config.medium_trade_ratio:
            return random.uniform(0.1, 1.0)   # Medium trades
        else:
            return random.uniform(1.0, 5.0)   # Large trades

    def _is_peak_hours(self) -> bool:
        """Check if current time is during peak trading hours"""
        if not self.config.trending_config.peak_hours_only:
            return True
            
        current_hour = datetime.utcnow().hour
        # Peak hours: 14:00-20:00 UTC (US morning/afternoon)
        return 14 <= current_hour <= 20

    async def _pump_fakeout_pattern(self, duration_minutes: int = 60):
        """Rapid buys followed by slow distribution to mimic organic pumps"""
        logger.info("🔄 Starting Pump-Fakeout volume pattern")
        wallets = await load_wallets(self.config)
        if not wallets:
            return

        end_time = time.time() + duration_minutes * 60
        phase = "accumulation"
        
        while self.is_running and time.time() < end_time:
            wallet = random.choice(wallets)
            
            if phase == "accumulation":
                # Rapid buying phase
                amount = self._get_trade_size_by_distribution() * 1.5  # Larger buys
                success = await self.amm.execute_trade(wallet, TradeType.BUY, amount)
                delay = random.uniform(5, 15)  # Quick succession
                
                # Switch to distribution after 40% of time
                if time.time() > end_time - (duration_minutes * 60 * 0.6):
                    phase = "distribution"
                    
            else:  # distribution phase
                # Slower, mixed trading
                if random.random() < 0.7:  # 70% sells
                    amount = self._get_trade_size_by_distribution() * 0.7  # Smaller sells
                    success = await self.amm.execute_trade(wallet, TradeType.SELL, amount)
                else:
                    amount = self._get_trade_size_by_distribution()
                    success = await self.amm.execute_trade(wallet, TradeType.BUY, amount)
                delay = random.uniform(20, 45)  # Slower pace
                
            if success:
                self.trending_metrics['total_volume_generated'] += amount
                
            await asyncio.sleep(delay)

    async def _steady_accumulation_pattern(self, duration_minutes: int = 90):
        """Consistent buy pressure with occasional profit-taking"""
        logger.info("📈 Starting Steady Accumulation volume pattern")
        wallets = await load_wallets(self.config)
        if not wallets:
            return

        end_time = time.time() + duration_minutes * 60
        
        while self.is_running and time.time() < end_time:
            wallet = random.choice(wallets)
            
            # 80% buys, 20% sells to create upward pressure
            if random.random() < 0.8:
                trade_type = TradeType.BUY
                amount = self._get_trade_size_by_distribution()
            else:
                trade_type = TradeType.SELL
                amount = self._get_trade_size_by_distribution() * 0.5  # Smaller sells
                
            success = await self.amm.execute_trade(wallet, trade_type, amount)
            
            if success:
                self.trending_metrics['total_volume_generated'] += amount
                
            delay = random.uniform(15, 30)
            await asyncio.sleep(delay)

    async def _whale_imitation_pattern(self, duration_minutes: int = 45):
        """Large trades to attract attention and create visible market impact"""
        logger.info("🐋 Starting Whale Imitation volume pattern")
        wallets = await load_wallets(self.config)
        if not wallets:
            return

        end_time = time.time() + duration_minutes * 60
        
        while self.is_running and time.time() < end_time:
            wallet = random.choice(wallets)
            
            # Larger trade sizes for whale imitation
            amount = random.uniform(2.0, 8.0)
            
            # Mix of buys and sells but mostly buys
            if random.random() < 0.7:
                trade_type = TradeType.BUY
            else:
                trade_type = TradeType.SELL
                
            success = await self.amm.execute_trade(wallet, trade_type, amount)
            
            if success:
                self.trending_metrics['total_volume_generated'] += amount
                
            # Longer delays between large trades
            delay = random.uniform(45, 120)
            await asyncio.sleep(delay)

    async def run_trending_optimized_volume(self, duration_minutes: int = 180):
        """Execute trending-optimized volume patterns"""
        logger.info(f"🚀 Starting Trending-Optimized Volume for {duration_minutes} minutes")
        
        if not self._is_peak_hours():
            logger.warning("⏰ Not in peak hours. Consider running during 14:00-20:00 UTC for best results")
            
        patterns = [
            self._pump_fakeout_pattern,
            self._steady_accumulation_pattern, 
            self._whale_imitation_pattern
        ]
        
        pattern_duration = duration_minutes // len(patterns)
        
        for pattern in patterns:
            if not self.is_running:
                break
            await pattern(pattern_duration)
            logger.info(f"✅ Completed {pattern.__name__}")
            
        logger.info(f"✅ Trending volume completed. Total volume: {self.trending_metrics['total_volume_generated']:.2f} SOL")

    async def advanced_holder_simulation(self, target_holders: int = None):
        """Create organic-looking holder growth with staggered timing"""
        if target_holders is None:
            target_holders = self.config.trending_config.unique_holders_target
            
        logger.info(f"👥 Starting Advanced Holder Simulation targeting {target_holders} holders")
        
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available for holder simulation")
            return

        # Use subset of wallets for holder simulation
        holder_wallets = wallets[:min(target_holders, len(wallets))]
        
        for i, wallet in enumerate(holder_wallets):
            if not self.is_running:
                break
                
            try:
                # Vary amounts to simulate different investor sizes
                if i < target_holders * 0.1:  # Top 10% as "whales"
                    amount = random.uniform(1.0, 5.0)
                elif i < target_holders * 0.4:  # Next 30% as medium
                    amount = random.uniform(0.1, 1.0)
                else:  # Remaining 60% as small holders
                    amount = random.uniform(0.01, 0.1)
                    
                input_mint = "So11111111111111111111111111111111111111112"
                output_mint = self.config.token_address
                swap_amount = int(amount * 1e9)
                
                quote = await self.jupiter_client.get_quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=swap_amount,
                    slippage_bps=100
                )
                
                if 'error' not in quote:
                    swap_data = await self.jupiter_client.swap(quote, wallet)
                    if 'error' not in swap_data:
                        success = await self.jupiter_client.execute_swap_transaction(swap_data, wallet)
                        if success:
                            logger.info(f"Holder {i+1}/{len(holder_wallets)} added: {amount:.4f} SOL -> {self.config.token_address}")
                            self.trending_metrics['unique_wallets_used'] += 1
                
                # Stagger acquisitions to look organic
                delay = random.uniform(30, 120)  # 30s to 2 minutes between acquisitions
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Holder simulation failed for wallet {i}: {e}")
                
        logger.info("✅ Advanced holder simulation completed")

    async def run_trending_optimized_auto(self):
        """Complete trending-optimized strategy"""
        logger.info("🎯 Starting Complete Trending-Optimized Strategy")
        self.trending_metrics['trending_start_time'] = time.time()
        
        # Phase 1: Build Holder Foundation
        logger.info("📊 Phase 1: Building Holder Foundation")
        await self.advanced_holder_simulation()
        
        # Phase 2: Volume Ramp-up
        logger.info("📈 Phase 2: Strategic Volume Ramp-up")
        await self.run_trending_optimized_volume(self.config.trending_config.trending_duration_hours * 60)
        
        # Phase 3: Sustained Market Making
        logger.info("🔄 Phase 3: Sustained Market Making")
        await self.run_hybrid_market_making(4)  # 4 hours of sustained MM
        
        logger.info("✅ Trending optimization completed successfully!")
        self._log_trending_performance()

    def _log_trending_performance(self):
        """Log trending-specific performance metrics"""
        runtime = time.time() - self.trending_metrics['trending_start_time']
        logger.info("=" * 60)
        logger.info("TRENDING PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"Total Volume Generated: {self.trending_metrics['total_volume_generated']:.2f} SOL")
        logger.info(f"Unique Wallets Used: {self.trending_metrics['unique_wallets_used']}")
        logger.info(f"Runtime: {runtime/3600:.2f} hours")
        logger.info(f"Volume/Hour: {self.trending_metrics['total_volume_generated']/(runtime/3600):.2f} SOL")
        logger.info("=" * 60)

    # Core Trading Methods
    async def run_hybrid_market_making(self, duration_hours: int = 24):
        logger.info(f"Starting hybrid market making for {duration_hours} hours")
        self.is_running = True
        end_time = time.time() + duration_hours * 3600
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available for trading")
            return
        iteration = 0
        while self.is_running and time.time() < end_time:
            try:
                wallet = random.choice(wallets)
                if random.random() < self.config.ai_parameters.buy_sell_ratio:
                    trade_type = TradeType.BUY
                else:
                    trade_type = TradeType.SELL
                if self.config.ai_parameters.use_market_context:
                    amount = self._generate_adaptive_amount(trade_type)
                else:
                    amount = random.uniform(
                        self.config.ai_parameters.min_trade_amount,
                        self.config.ai_parameters.max_trade_amount
                    )
                success = await self.amm.execute_trade(wallet, trade_type, amount)
                delay = self._calculate_adaptive_delay()
                await asyncio.sleep(delay)
                iteration += 1
                if iteration % 20 == 0:
                    self._log_performance()
            except Exception as e:
                logger.error(f"Trading iteration failed: {e}")
                await asyncio.sleep(5)
        logger.info("Hybrid market making completed")
        self._log_final_performance()

    async def run_volume_boosting(self, token: str, use_random_amount: bool, amount: float, min_amount: float,
                                 max_amount: float, frequency: int, duration: int):
        if use_random_amount and (min_amount <= 0 or max_amount <= 0 or min_amount > max_amount):
            logger.error("Invalid random amount range for volume boosting")
            raise ValueError("min_amount and max_amount must be positive and min_amount <= max_amount")
        if not use_random_amount and amount <= 0:
            logger.error("Invalid fixed amount for volume boosting")
            raise ValueError("amount must be positive when use_random_amount is False")
        logger.info(f"Starting volume boosting for {token} for {duration} minutes")
        self.is_running = True
        end_time = time.time() + duration * 60
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available for trading")
            return
        iteration = 0
        while self.is_running and time.time() < end_time:
            for wallet in wallets:
                if not self.is_running or time.time() >= end_time:
                    break
                try:
                    trade_amount = random.uniform(min_amount, max_amount) if use_random_amount else amount
                    input_mint = "So11111111111111111111111111111111111111112"
                    output_mint = token
                    swap_amount = int(trade_amount * 1e9)
                    quote = await self.jupiter_client.get_quote(
                        input_mint=input_mint,
                        output_mint=output_mint,
                        amount=swap_amount,
                        slippage_bps=100
                    )
                    if 'error' not in quote:
                        swap_data = await self.jupiter_client.swap(quote, wallet)
                        if 'error' not in swap_data:
                            success = await self.jupiter_client.execute_swap_transaction(swap_data, wallet)
                            if success:
                                logger.info(f"Volume boost swap executed: {trade_amount} SOL -> {token}")
                    delay = 60.0 / frequency
                    await asyncio.sleep(delay)
                    iteration += 1
                except Exception as e:
                    logger.error(f"Volume boost iteration failed: {e}")
                    await asyncio.sleep(5)
        logger.info("Volume boosting completed")

    async def add_holders(self, count: int, min_amount: float, max_amount: float):
        logger.info(f"Starting holder simulation for {count} wallets with amounts {min_amount}-{max_amount} SOL")
        self.is_running = True
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available for holder simulation")
            return
        selected_wallets = random.sample(wallets, min(count, len(wallets)))
        for i, wallet in enumerate(selected_wallets, 1):
            if not self.is_running:
                break
            try:
                amount = random.uniform(min_amount, max_amount)
                input_mint = "So11111111111111111111111111111111111111112"
                output_mint = self.config.token_address
                swap_amount = int(amount * 1e9)
                quote = await self.jupiter_client.get_quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=swap_amount,
                    slippage_bps=100
                )
                if 'error' not in quote:
                    swap_data = await self.jupiter_client.swap(quote, wallet)
                    if 'error' not in swap_data:
                        success = await self.jupiter_client.execute_swap_transaction(swap_data, wallet)
                        if success:
                            logger.info(
                                f"Holder {i}/{len(selected_wallets)} added: {amount:.4f} SOL -> {self.config.token_address}")
                delay = random.uniform(self.config.ai_parameters.min_delay, self.config.ai_parameters.max_delay)
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Holder simulation failed for wallet {i}: {e}")
        logger.info("Holder simulation completed")

    async def withdraw_balances(self, token_mint: Optional[str], amount: Optional[float], decimals: int = 9):
        logger.info(f"Starting balance withdrawal to main wallet: {self.config.main_wallet_pubkey}")
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available for withdrawal")
            return
        to_pubkey = Pubkey.from_string(self.config.main_wallet_pubkey)
        async with AsyncClient(self.config.rpc_url) as client:
            for i, wallet in enumerate(wallets, 1):
                try:
                    if token_mint:
                        ata = await get_associated_token_account(wallet.pubkey(), token_mint, wallet, client)
                        if not ata:
                            logger.warning(f"Wallet {i} has no ATA for token {token_mint}")
                            continue
                        balance_resp = await client.get_token_account_balance(Pubkey.from_string(ata))
                        if not balance_resp.value:
                            logger.warning(f"Wallet {i} has no balance for token {token_mint}")
                            continue
                        balance = balance_resp.value.amount
                        amount_to_transfer = int(amount * (10 ** decimals)) if amount else balance
                        if amount_to_transfer <= 0:
                            logger.warning(f"Wallet {i} has insufficient token balance")
                            continue
                        success = await transfer_spl_token(
                            wallet, to_pubkey, token_mint, amount_to_transfer, decimals, self.config.rpc_url
                        )
                        if success:
                            logger.info(
                                f"Withdrew {amount_to_transfer / (10 ** decimals)} tokens from wallet {i} to main wallet")
                    else:
                        balance_resp = await client.get_balance(wallet.pubkey())
                        if not balance_resp.value:
                            logger.warning(f"Wallet {i} has no SOL balance")
                            continue
                        balance = balance_resp.value
                        amount_to_transfer = int(amount * 1e9) if amount else max(0, balance - 1000000)
                        if amount_to_transfer <= 0:
                            logger.warning(f"Wallet {i} has insufficient SOL balance")
                            continue
                        success = await transfer_sol(wallet, to_pubkey, amount_to_transfer, self.config.rpc_url)
                        if success:
                            logger.info(f"Withdrew {amount_to_transfer / 1e9} SOL from wallet {i} to main wallet")
                    delay = random.uniform(self.config.ai_parameters.min_delay, self.config.ai_parameters.max_delay)
                    await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Withdrawal failed for wallet {i}: {e}")
        logger.info("Balance withdrawal completed")

    async def run_all_auto(self):
        logger.info("Starting all-auto strategy: Holder Simulation -> Volume Boosting -> Hybrid Market Making")
        self.is_running = True

        # Validate configurations
        if not self.config.token_address:
            logger.error("Token address not set")
            raise ValueError("Please set STEALTH_TOKEN_ADDRESS in .env or token_address in config.yaml")
        if self.config.holder_config.count <= 0:
            logger.error("Invalid holder count")
            raise ValueError("Please set a positive STEALTH_HOLDER_COUNT in .env or holder_config.count in config.yaml")
        if self.config.holder_config.min_amount <= 0 or self.config.holder_config.max_amount <= 0 or \
                self.config.holder_config.min_amount > self.config.holder_config.max_amount:
            logger.error("Invalid holder amount range")
            raise ValueError("Please set valid STEALTH_HOLDER_MIN_AMOUNT and STEALTH_HOLDER_MAX_AMOUNT in .env or holder_config in config.yaml")
        if not self.config.volume_boost_config.token:
            logger.error("Volume boost token not set")
            raise ValueError("Please set STEALTH_VOLUME_BOOST_TOKEN in .env or volume_boost_config.token in config.yaml")
        if not self.config.volume_boost_config.use_random_amount and self.config.volume_boost_config.amount <= 0:
            logger.error("Invalid volume boost amount")
            raise ValueError("Please set a positive STEALTH_VOLUME_BOOST_AMOUNT in .env or volume_boost_config.amount in config.yaml")
        if self.config.volume_boost_config.use_random_amount and (
                self.config.volume_boost_config.min_amount <= 0 or
                self.config.volume_boost_config.max_amount <= 0 or
                self.config.volume_boost_config.min_amount > self.config.volume_boost_config.max_amount):
            logger.error("Invalid volume boost random amount range")
            raise ValueError("Please set valid STEALTH_VOLUME_BOOST_MIN_AMOUNT and STEALTH_VOLUME_BOOST_MAX_AMOUNT in .env or volume_boost_config in config.yaml")
        if self.config.volume_boost_config.frequency <= 0:
            logger.error("Invalid volume boost frequency")
            raise ValueError("Please set a positive STEALTH_VOLUME_BOOST_FREQUENCY in .env or volume_boost_config.frequency in config.yaml")
        if self.config.volume_boost_config.duration <= 0:
            logger.error("Invalid volume boost duration")
            raise ValueError("Please set a positive STEALTH_VOLUME_BOOST_DURATION in .env or volume_boost_config.duration in config.yaml")
        if self.config.hybrid_mm_config.duration_hours <= 0:
            logger.error("Invalid hybrid MM duration")
            raise ValueError("Please set a positive STEALTH_HYBRID_MM_DURATION_HOURS in .env or hybrid_mm_config.duration_hours in config.yaml")

        # Step 1: Holder Simulation
        try:
            logger.info("Executing Holder Simulation...")
            await self.add_holders(
                self.config.holder_config.count,
                self.config.holder_config.min_amount,
                self.config.holder_config.max_amount
            )
            logger.info("Holder Simulation completed successfully")
        except Exception as e:
            logger.error(f"Holder Simulation failed: {e}")
            self.is_running = False
            return

        # Step 2: Volume Boosting
        try:
            logger.info("Executing Volume Boosting...")
            await self.run_volume_boosting(
                self.config.volume_boost_config.token,
                self.config.volume_boost_config.use_random_amount,
                self.config.volume_boost_config.amount,
                self.config.volume_boost_config.min_amount,
                self.config.volume_boost_config.max_amount,
                self.config.volume_boost_config.frequency,
                self.config.volume_boost_config.duration
            )
            logger.info("Volume Boosting completed successfully")
        except Exception as e:
            logger.error(f"Volume Boosting failed: {e}")
            self.is_running = False
            return

        # Step 3: Hybrid Market Making
        try:
            logger.info("Executing Hybrid Market Making...")
            await self.run_hybrid_market_making(self.config.hybrid_mm_config.duration_hours)
            logger.info("Hybrid Market Making completed successfully")
        except Exception as e:
            logger.error(f"Hybrid Market Making failed: {e}")
            self.is_running = False
            return

        logger.info("All-auto strategy completed")

    def _generate_adaptive_amount(self, trade_type: TradeType) -> float:
        base_amount = random.uniform(
            self.config.ai_parameters.min_trade_amount,
            self.config.ai_parameters.max_trade_amount
        )
        if self.amm.inventory['base_token'] > 0:
            inventory_ratio = self.amm.inventory['base_token'] / (
                        self.amm.inventory['base_token'] + self.amm.inventory['quote_token'] / self.amm.current_price)
        else:
            inventory_ratio = 0.5
        if trade_type == TradeType.BUY and inventory_ratio < 0.3:
            base_amount *= 1.5
        elif trade_type == TradeType.SELL and inventory_ratio > 0.7:
            base_amount *= 1.5
        return min(base_amount, self.config.ai_parameters.max_trade_amount)

    def _calculate_adaptive_delay(self) -> float:
        base_delay = random.uniform(
            self.config.ai_parameters.min_delay,
            self.config.ai_parameters.max_delay
        )
        recent_trades = [t for t in self.amm.trade_history if time.time() - t['timestamp'] < 300]
        if len(recent_trades) > 10:
            base_delay *= 1.5
        return base_delay

    def _log_performance(self):
        metrics = self.amm.get_performance_metrics()
        logger.info(
            f"Performance: {metrics['total_trades']} trades, "
            f"{metrics['success_rate']:.1%} success, "
            f"${metrics['inventory_value']:.2f} inventory, "
            f"${metrics['total_volume']:.2f} volume"
        )

    def _log_final_performance(self):
        metrics = self.amm.get_performance_metrics()
        logger.info("=" * 50)
        logger.info("FINAL PERFORMANCE METRICS")
        logger.info("=" * 50)
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Successful Trades: {metrics['successful_trades']}")
        logger.info(f"Success Rate: {metrics['success_rate']:.1%}")
        logger.info(f"Total Volume: ${metrics['total_volume']:.2f}")
        logger.info(f"Inventory Value: ${metrics['inventory_value']:.2f}")
        logger.info(f"Runtime: {metrics['runtime_hours']:.1f} hours")
        logger.info("=" * 50)

    def stop(self):
        self.is_running = False
        logger.info("Trading engine stopping...")

# CLI Commands
@click.group()
@click.version_option(version='4.1.0')
def cli():
    """StealthVolume CLI - Advanced Solana Trading Bot"""
    pass

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def init(config_path: str):
    """Initialize configuration file"""
    config = ConfigManager.load_config('env', config_path)
    ConfigManager.save_config(config)
    logger.info(f"Configuration initialized at {config_path}")
    logger.info("Edit the configuration file and generate wallets before trading")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--count', default=10, help='Number of wallets to generate')
def generate_wallets(config_path: str, count: int):
    """Generate new trading wallets"""
    config = ConfigManager.load_config('yaml', config_path)
    asyncio.run(generate_wallets(count, config))
    logger.info(f"Generated {count} wallets")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--duration', default=24, help='Duration in hours')
def hybrid_mm(config_path: str, duration: int):
    """Run hybrid market making strategy"""
    config = ConfigManager.load_config('yaml', config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_hybrid_market_making(duration))
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Strategy failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def hybrid_mm_auto(config_path: str):
    """Run hybrid market making with auto configuration"""
    config = ConfigManager.load_config('yaml', config_path)
    if config.hybrid_mm_config.duration_hours <= 0:
        logger.error("Invalid duration in hybrid_mm_config")
        raise click.ClickException("Please set a positive STEALTH_HYBRID_MM_DURATION_HOURS in .env or hybrid_mm_config.duration_hours in config.yaml")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_hybrid_market_making(config.hybrid_mm_config.duration_hours))
    except KeyboardInterrupt:
        logger.info("Hybrid MM stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Hybrid MM failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--token', required=True, help='Token mint address')
@click.option('--use-random-amount', is_flag=True, help='Use random amounts for swaps')
@click.option('--amount', type=float, default=0.1, help='Fixed amount per swap in SOL')
@click.option('--min-amount', type=float, default=0.01, help='Minimum amount for random swaps in SOL')
@click.option('--max-amount', type=float, default=0.1, help='Maximum amount for random swaps in SOL')
@click.option('--frequency', type=int, default=10, help='Swaps per minute')
@click.option('--duration', type=int, default=60, help='Duration in minutes')
def boost_volume(config_path: str, token: str, use_random_amount: bool, amount: float, min_amount: float,
                 max_amount: float, frequency: int, duration: int):
    """Run volume boosting for specific token"""
    config = ConfigManager.load_config('yaml', config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_volume_boosting(token, use_random_amount, amount, min_amount, max_amount, frequency, duration))
    except KeyboardInterrupt:
        logger.info("Volume boosting stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Volume boosting failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def boost_volume_auto(config_path: str):
    """Run volume boosting with auto configuration"""
    config = ConfigManager.load_config('yaml', config_path)
    if not config.volume_boost_config.token:
        logger.error("Token mint address not set in volume_boost_config")
        raise click.ClickException("Please set STEALTH_VOLUME_BOOST_TOKEN in .env or volume_boost_config.token in config.yaml")
    if not config.volume_boost_config.use_random_amount and config.volume_boost_config.amount <= 0:
        logger.error("Invalid amount in volume_boost_config")
        raise click.ClickException("Please set a positive STEALTH_VOLUME_BOOST_AMOUNT in .env or volume_boost_config.amount in config.yaml")
    if config.volume_boost_config.use_random_amount and (
            config.volume_boost_config.min_amount <= 0 or
            config.volume_boost_config.max_amount <= 0 or
            config.volume_boost_config.min_amount > config.volume_boost_config.max_amount):
        logger.error("Invalid random amount range in volume_boost_config")
        raise click.ClickException(
            "Please set valid STEALTH_VOLUME_BOOST_MIN_AMOUNT and STEALTH_VOLUME_BOOST_MAX_AMOUNT in .env or volume_boost_config in config.yaml")
    if config.volume_boost_config.frequency <= 0:
        logger.error("Invalid frequency in volume_boost_config")
        raise click.ClickException("Please set a positive STEALTH_VOLUME_BOOST_FREQUENCY in .env or volume_boost_config.frequency in config.yaml")
    if config.volume_boost_config.duration <= 0:
        logger.error("Invalid duration in volume_boost_config")
        raise click.ClickException("Please set a positive STEALTH_VOLUME_BOOST_DURATION in .env or volume_boost_config.duration in config.yaml")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_volume_boosting(
            config.volume_boost_config.token,
            config.volume_boost_config.use_random_amount,
            config.volume_boost_config.amount,
            config.volume_boost_config.min_amount,
            config.volume_boost_config.max_amount,
            config.volume_boost_config.frequency,
            config.volume_boost_config.duration
        ))
    except KeyboardInterrupt:
        logger.info("Volume boosting stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Volume boosting failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--input-mint', required=True, help='Input token mint address')
@click.option('--output-mint', required=True, help='Output token mint address')
@click.option('--amount', type=float, required=True, help='Amount to swap')
@click.option('--slippage', type=float, default=1.0, help='Slippage percentage')
def swap(config_path: str, input_mint: str, output_mint: str, amount: float, slippage: float):
    """Execute a single swap"""
    config = ConfigManager.load_config('yaml', config_path)
    async def execute_swap():
        async with aiohttp.ClientSession() as session:
            jupiter = JupiterLiteClient(session, config)
            wallets = await load_wallets(config)
            if not wallets:
                logger.error("No wallets available")
                return
            wallet = wallets[0]
            amount_lamports = int(amount * 1e9)
            quote = await jupiter.get_quote(input_mint, output_mint, amount_lamports, int(slippage * 100))
            if 'error' in quote:
                logger.error(f"Quote failed: {quote['error']}")
                return
            swap_data = await jupiter.swap(quote, wallet)
            if 'error' in swap_data:
                logger.error(f"Swap failed: {swap_data['error']}")
                return
            success = await jupiter.execute_swap_transaction(swap_data, wallet)
            if success:
                logger.info("Swap executed successfully!")
            else:
                logger.error("Swap execution failed")
    asyncio.run(execute_swap())

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--count', type=int, default=50, help='Number of wallets to use for holder simulation')
@click.option('--min-amount', type=float, default=0.001, help='Minimum trade amount in SOL')
@click.option('--max-amount', type=float, default=0.05, help='Maximum trade amount in SOL')
def add_holders(config_path: str, count: int, min_amount: float, max_amount: float):
    """Add simulated holders"""
    config = ConfigManager.load_config('yaml', config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.add_holders(count, min_amount, max_amount))
    except KeyboardInterrupt:
        logger.info("Holder simulation stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Holder simulation failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def add_holders_auto(config_path: str):
    """Add holders with auto configuration"""
    config = ConfigManager.load_config('yaml', config_path)
    if config.holder_config.count <= 0:
        logger.error("Invalid count in holder_config")
        raise click.ClickException("Please set a positive STEALTH_HOLDER_COUNT in .env or holder_config.count in config.yaml")
    if config.holder_config.min_amount <= 0 or config.holder_config.max_amount <= 0 or config.holder_config.min_amount > config.holder_config.max_amount:
        logger.error("Invalid amounts in holder_config")
        raise click.ClickException("Please set valid STEALTH_HOLDER_MIN_AMOUNT and STEALTH_HOLDER_MAX_AMOUNT in .env or holder_config in config.yaml")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.add_holders(config.holder_config.count, config.holder_config.min_amount, config.holder_config.max_amount))
    except KeyboardInterrupt:
        logger.info("Holder simulation stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Holder simulation failed: {e}")

@cli.command()
def show_main_private():
    """Display main wallet private key (use with caution)"""
    private_key = base58.b58encode(MAIN_WALLET_KEYPAIR.to_bytes()).decode()
    logger.warning("WARNING: Displaying private key is insecure. Use only in safe environments.")
    print(f"Main Wallet Private Key (base58): {private_key}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--token-mint', default=None, help='Token mint address (omit for SOL)')
@click.option('--amount', type=float, default=None, help='Amount to withdraw (omit for full balance)')
@click.option('--decimals', type=int, default=9, help='Token decimals (for SPL tokens)')
def withdraw(config_path: str, token_mint: Optional[str], amount: Optional[float], decimals: int):
    """Withdraw balances to main wallet"""
    config = ConfigManager.load_config('yaml', config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.withdraw_balances(token_mint, amount, decimals))
    except KeyboardInterrupt:
        logger.info("Withdrawal stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Withdrawal failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def withdraw_auto(config_path: str):
    """Withdraw with auto configuration"""
    config = ConfigManager.load_config('yaml', config_path)
    if config.withdraw_config.amount is not None and config.withdraw_config.amount < 0:
        logger.error("Invalid amount in withdraw_config")
        raise click.ClickException("Please set a non-negative STEALTH_WITHDRAW_AMOUNT in .env or withdraw_config.amount in config.yaml")
    if config.withdraw_config.decimals <= 0:
        logger.error("Invalid decimals in withdraw_config")
        raise click.ClickException("Please set a positive STEALTH_WITHDRAW_DECIMALS in .env or withdraw_config.decimals in config.yaml")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.withdraw_balances(
            config.withdraw_config.token_mint,
            config.withdraw_config.amount,
            config.withdraw_config.decimals
        ))
    except KeyboardInterrupt:
        logger.info("Withdrawal stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Withdrawal failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def all_auto(config_path: str):
    """Run complete automated strategy"""
    config = ConfigManager.load_config('yaml', config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_all_auto())
    except KeyboardInterrupt:
        logger.info("All-auto strategy stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"All-auto strategy failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def trending_optimized(config_path: str):
    """Run trending-optimized strategy for maximum DEX visibility"""
    config = ConfigManager.load_config('yaml', config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_trending_optimized_auto())
    except KeyboardInterrupt:
        logger.info("Trending strategy stopped by user")
        engine.stop()
    except Exception as e:
        logger.error(f"Trending strategy failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def status(config_path: str):
    """Display current bot status and configuration"""
    config = ConfigManager.load_config('yaml', config_path)
    logger.info("=" * 60)
    logger.info("STEALTHVOLUME STATUS - v4.1.0")
    logger.info("=" * 60)
    logger.info(f"RPC URL: {config.rpc_url}")
    logger.info(f"Token Address: {config.token_address}")
    logger.info(f"DEXes: {', '.join(config.dexes)}")
    logger.info(f"Main Wallet: {config.main_wallet_pubkey}")
    logger.info(f"Trading Mode: {config.trading_mode}")
    logger.info(f"Wallets Loaded: {len(config.wallets)}")
    
    logger.info("\n📊 TRENDING CONFIG:")
    logger.info(f"  Raydium Threshold: {config.trending_config.raydium_volume_threshold} USD")
    logger.info(f"  Orca Threshold: {config.trending_config.orca_volume_threshold} USD")
    logger.info(f"  Holders Target: {config.trending_config.unique_holders_target}")
    logger.info(f"  Duration: {config.trending_config.trending_duration_hours} hours")
    logger.info(f"  Peak Hours Only: {config.trending_config.peak_hours_only}")
    logger.info(f"  Trade Distribution: {config.trending_config.small_trade_ratio*100:.0f}% small, "
                f"{config.trending_config.medium_trade_ratio*100:.0f}% medium, "
                f"{config.trending_config.large_trade_ratio*100:.0f}% large")
    
    logger.info("\n⚡ VOLUME BOOST:")
    logger.info(f"  Token: {config.volume_boost_config.token}")
    logger.info(f"  Random Amounts: {config.volume_boost_config.use_random_amount}")
    if config.volume_boost_config.use_random_amount:
        logger.info(f"  Amount Range: {config.volume_boost_config.min_amount}-{config.volume_boost_config.max_amount} SOL")
    else:
        logger.info(f"  Fixed Amount: {config.volume_boost_config.amount} SOL")
    logger.info(f"  Frequency: {config.volume_boost_config.frequency}/min")
    logger.info(f"  Duration: {config.volume_boost_config.duration} min")
    
    logger.info("\n👥 HOLDER SIMULATION:")
    logger.info(f"  Count: {config.holder_config.count}")
    logger.info(f"  Amount Range: {config.holder_config.min_amount}-{config.holder_config.max_amount} SOL")
    
    logger.info("\n🤖 HYBRID MARKET MAKING:")
    logger.info(f"  Duration: {config.hybrid_mm_config.duration_hours} hours")
    logger.info(f"  Buy/Sell Ratio: {config.ai_parameters.buy_sell_ratio}")
    logger.info(f"  Trade Range: {config.ai_parameters.min_trade_amount}-{config.ai_parameters.max_trade_amount} SOL")
    
    logger.info("=" * 60)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 STEALTHVOLUME CLI v4.1.0 - TRENDING OPTIMIZED")
    print("=" * 70)
    print("FEATURES:")
    print("  - Jupiter Lite API Integration")
    print("  - Advanced AMM with Hybrid Pricing") 
    print("  - Trending Optimization for DEX Visibility")
    print("  - Multi-Strategy Trading Engine")
    print("  - Holder Simulation & Volume Boosting")
    print("  - Balance Withdrawal & Security")
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

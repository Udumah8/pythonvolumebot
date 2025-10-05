#!/usr/bin/env python3
"""
StealthVolume CLI - Jupiter Lite API Integration with Advanced AMM
Production-ready implementation for Solana token trading with holder simulation, volume boosting, and market making
"""

import asyncio
import base64
import json
import logging
import random
import time
import yaml
import aiohttp
import click
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.transaction import VersionedTransaction
from solders.system_program import TransferParams, transfer
from spl.token.instructions import transfer_checked, TransferCheckedParams
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.client import Token
import base58
from cryptography.fernet import Fernet

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

@dataclass
class AIParameters:
    buy_sell_ratio: float = 0.7
    use_market_context: bool = True
    min_trade_amount: float = 0.01
    max_trade_amount: float = 5.0
    min_delay: float = 1.0
    max_delay: float = 300.0
    max_slippage: float = 5.0

@dataclass
class AMMConfig:
    strategy: str = "hybrid"
    base_token: str = ""
    quote_token: str = "So11111111111111111111111111111111111111112"
    initial_price: float = 0.01
    liquidity_depth: float = 5000.0
    max_slippage: float = 2.0
    fee_rate: float = 0.003
    min_trade_size: float = 0.1
    max_trade_size: float = 10.0
    spread_target: float = 0.02

@dataclass
class SecurityConfig:
    randomize_timing: bool = True
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.2
    enable_encryption: bool = True

@dataclass
class JupiterConfig:
    base_url: str = "https://lite-api.jup.ag"
    swap_version: str = "v1"
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
    config_path: str
    main_wallet_pubkey: str
    volume_boost_config: VolumeBoostConfig = field(default_factory=VolumeBoostConfig)
    holder_config: HolderConfig = field(default_factory=HolderConfig)
    hybrid_mm_config: HybridMMConfig = field(default_factory=HybridMMConfig)
    funding_config: FundingConfig = field(default_factory=FundingConfig)
    withdraw_config: WithdrawConfig = field(default_factory=WithdrawConfig)
    ai_parameters: AIParameters = field(default_factory=AIParameters)
    amm_config: AMMConfig = field(default_factory=AMMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    jupiter_config: JupiterConfig = field(default_factory=JupiterConfig)

# Configuration Management
class ConfigManager:
    @staticmethod
    def load_config(config_path: str = 'config.yaml') -> TradeConfig:
        load_dotenv()
        encryption_key = Fernet.generate_key()
        config = TradeConfig(
            rpc_url=os.getenv('STEALTH_RPC_URL', 'https://api.mainnet-beta.solana.com'),
            jupiter_api_key=os.getenv('STEALTH_JUPITER_API_KEY', ''),
            token_address=os.getenv('STEALTH_TOKEN_ADDRESS', ''),
            dexes=os.getenv('STEALTH_DEXES', 'Raydium,Orca').split(','),
            wallets=[],
            encryption_key=encryption_key,
            config_path=config_path,
            main_wallet_pubkey=MAIN_WALLET_PUBKEY,
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
                amount=float(os.getenv('STEALTH_WITHDRAW_AMOUNT', '0')) if os.getenv('STEALTH_WITHDRAW_AMOUNT') else None,
                decimals=int(os.getenv('STEALTH_WITHDRAW_DECIMALS', '9'))
            )
        )
        if Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                config.wallets = yaml_config.get('wallets', [])
                if yaml_config.get('encryption_key'):
                    config.encryption_key = base58.b58decode(yaml_config['encryption_key'])
            except Exception as e:
                logger.warning(f"Failed to load YAML config: {e}, using .env")
        return config

    @staticmethod
    def save_config(config: TradeConfig):
        config_dict = {
            'rpc_url': config.rpc_url,
            'jupiter_api_key': config.jupiter_api_key,
            'token_address': config.token_address,
            'dexes': config.dexes,
            'wallets': config.wallets,
            'encryption_key': base58.b58encode(config.encryption_key).decode(),
            'config_path': config.config_path,
            'main_wallet_pubkey': config.main_wallet_pubkey,
            'volume_boost_config': {
                'token': config.volume_boost_config.token,
                'use_random_amount': config.volume_boost_config.use_random_amount,
                'amount': config.volume_boost_config.amount,
                'min_amount': config.volume_boost_config.min_amount,
                'max_amount': config.volume_boost_config.max_amount,
                'frequency': config.volume_boost_config.frequency,
                'duration': config.volume_boost_config.duration
            },
            'holder_config': {
                'count': config.holder_config.count,
                'min_amount': config.holder_config.min_amount,
                'max_amount': config.holder_config.max_amount
            },
            'hybrid_mm_config': {
                'duration_hours': config.hybrid_mm_config.duration_hours
            },
            'funding_config': {
                'enable_auto_funding': config.funding_config.enable_auto_funding,
                'min_fund_amount': config.funding_config.min_fund_amount,
                'max_fund_amount': config.funding_config.max_fund_amount
            },
            'withdraw_config': {
                'token_mint': config.withdraw_config.token_mint,
                'amount': config.withdraw_config.amount,
                'decimals': config.withdraw_config.decimals
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
        secret_key = wallet.secret()
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
            private_key = decrypt_data(wallet_data['private_key'], config.encryption_key)
            secret_key = base58.b58decode(private_key)
            wallet = Keypair.from_seed(secret_key)
            wallets.append(wallet)
        except Exception as e:
            logger.error(f"Failed to load wallet {wallet_data.get('public_key', 'unknown')}: {e}")
    logger.info(f"Loaded {len(wallets)} wallets")
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
        elapsed = current_time - self.last_request_time
        if elapsed < self.config.security.rate_limit_delay:
            await asyncio.sleep(self.config.security.rate_limit_delay - elapsed)
        self.last_request_time = current_time
        self.request_count += 1

    async def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 100) -> Dict[str, Any]:
        await self._rate_limit()
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps),
                'swapMode': 'ExactIn'
            }
            url = f"{self.jupiter_config.base_url}/swap/{self.jupiter_config.swap_version}/quote"
            timeout = aiohttp.ClientTimeout(total=self.jupiter_config.quote_timeout)
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    quote_data = await response.json()
                    logger.debug(f"Quote obtained: {float(quote_data.get('outAmount', 0)) / 1e9:.6f} output")
                    return quote_data
                else:
                    error_text = await response.text()
                    logger.error(f"Quote error {response.status}: {error_text}")
                    return {'error': f'HTTP {response.status}: {error_text}'}
        except Exception as e:
            logger.error(f"Quote request failed: {e}")
            return {'error': str(e)}

    async def swap(self, quote_response: Dict[str, Any], wallet: Keypair) -> Dict[str, Any]:
        await self._rate_limit()
        try:
            payload = {
                'quoteResponse': quote_response,
                'userPublicKey': str(wallet.pubkey()),
                'wrapAndUnwrapSol': True,
                'dynamicComputeUnitLimit': True,
                'prioritizationFeeLamports': 'auto'
            }
            headers = {'Content-Type': 'application/json'}
            url = f"{self.jupiter_config.base_url}/swap/{self.jupiter_config.swap_version}/swap"
            timeout = aiohttp.ClientTimeout(total=self.jupiter_config.swap_timeout)
            async with self.session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Swap error {response.status}: {error_text}")
                    return {'error': f'HTTP {response.status}: {error_text}'}
        except Exception as e:
            logger.error(f"Swap failed: {e}")
            return {'error': str(e)}

    async def execute_swap_transaction(self, swap_data: Dict[str, Any], wallet: Keypair) -> bool:
        try:
            if 'error' in swap_data:
                logger.error(f"Cannot execute swap: {swap_data['error']}")
                return False
            swap_transaction = swap_data.get('swapTransaction')
            if not swap_transaction:
                logger.error("No swap transaction in response")
                return False
            tx_bytes = base64.b64decode(swap_transaction)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            async with AsyncClient(self.config.rpc_url) as client:
                balance_resp = await client.get_balance(wallet.pubkey())
                if not balance_resp.value or balance_resp.value < 1000000:
                    logger.error(f"Wallet {wallet.pubkey()} has insufficient SOL")
                    return False
                result = await client.send_transaction(transaction, wallet)
                if result.value:
                    logger.info(f"Swap transaction sent: {result.value}")
                    return True
                logger.error("Failed to send transaction")
                return False
        except Exception as e:
            logger.error(f"Transaction execution error: {e}")
            return False

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
                logger.error("Failed to get blockhash")
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
            logger.error("SOL transfer failed")
            return False
    except Exception as e:
        logger.error(f"SOL transfer error: {e}")
        return False

async def transfer_spl_token(wallet: Keypair, to_pubkey: Pubkey, token_mint: str, amount: int, decimals: int, rpc_url: str) -> bool:
    try:
        async with AsyncClient(rpc_url) as client:
            token = Token(client, Pubkey.from_string(token_mint), TOKEN_PROGRAM_ID, wallet)
            from_ata = await token.get_or_create_associated_account_info(wallet.pubkey())
            to_ata = await token.get_or_create_associated_account_info(to_pubkey)
            balance_resp = await client.get_token_account_balance(from_ata.address)
            if not balance_resp.value or balance_resp.value.amount < amount:
                logger.error(f"Wallet {wallet.pubkey()} has insufficient token balance")
                return False
            blockhash_resp = await client.get_latest_blockhash()
            if not blockhash_resp.value:
                logger.error("Failed to get blockhash")
                return False
            blockhash = blockhash_resp.value.blockhash
            transfer_params = TransferCheckedParams(
                program_id=TOKEN_PROGRAM_ID,
                source=from_ata.address,
                mint=Pubkey.from_string(token_mint),
                dest=to_ata.address,
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
            logger.error("SPL token transfer failed")
            return False
    except Exception as e:
        logger.error(f"SPL token transfer error: {e}")
        return False

# Advanced AMM
class AdvancedAMM:
    def __init__(self, config: TradeConfig, jupiter_client: JupiterLiteClient):
        self.config = config
        self.jupiter = jupiter_client
        self.inventory = {
            'base_token': config.amm_config.base_token_reserve,
            'quote_token': config.amm_config.quote_token_reserve
        }
        self.current_price = config.amm_config.initial_price
        self.trade_history = []

    async def execute_trade(self, wallet: Keypair, trade_type: TradeType, amount: float) -> bool:
        try:
            input_mint = "So11111111111111111111111111111111111111112" if trade_type == TradeType.BUY else self.config.token_address
            output_mint = self.config.token_address if trade_type == TradeType.BUY else "So11111111111111111111111111111111111111112"
            swap_amount = int(amount * 1e9)
            quote = await self.jupiter.get_quote(input_mint, output_mint, swap_amount, int(self.config.amm_config.max_slippage * 100))
            if 'error' in quote:
                logger.error(f"Quote failed: {quote['error']}")
                return False
            swap_data = await self.jupiter.swap(quote, wallet)
            if 'error' in swap_data:
                logger.error(f"Swap failed: {swap_data['error']}")
                return False
            success = await self.jupiter.execute_swap_transaction(swap_data, wallet)
            if success:
                self._update_inventory(trade_type, amount, float(quote.get('outAmount', 0)) / 1e9 / amount)
                logger.info(f"{trade_type.value} executed: {amount:.4f} at price {self.current_price:.6f}")
            return success
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _update_inventory(self, trade_type: TradeType, amount: float, price: float):
        if trade_type == TradeType.BUY:
            self.inventory['quote_token'] -= amount
            self.inventory['base_token'] += amount / price
        else:
            self.inventory['quote_token'] += amount * price
            self.inventory['base_token'] -= amount
        if self.inventory['base_token'] > 0:
            self.current_price = self.inventory['quote_token'] / self.inventory['base_token']
        self.trade_history.append({
            'timestamp': time.time(),
            'type': trade_type.value,
            'amount': amount,
            'price': self.current_price
        })

# Trading Engine
class StealthVolumeEngine:
    def __init__(self, config: TradeConfig):
        self.config = config
        self.jupiter_client = None
        self.amm = None
        self.is_running = False

    async def initialize(self):
        self.jupiter_client = JupiterLiteClient(aiohttp.ClientSession(), self.config)
        self.amm = AdvancedAMM(self.config, self.jupiter_client)
        logger.info("StealthVolume engine initialized")

    async def run_hybrid_market_making(self, duration_hours: int):
        logger.info(f"Starting hybrid market making for {duration_hours} hours")
        self.is_running = True
        end_time = time.time() + duration_hours * 3600
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available")
            return
        while self.is_running and time.time() < end_time:
            try:
                wallet = random.choice(wallets)
                trade_type = TradeType.BUY if random.random() < self.config.ai_parameters.buy_sell_ratio else TradeType.SELL
                amount = random.uniform(self.config.ai_parameters.min_trade_amount, self.config.ai_parameters.max_trade_amount)
                if self.config.ai_parameters.use_market_context and len(self.amm.trade_history) > 10:
                    amount = self._adjust_amount_based_on_context(trade_type, amount)
                success = await self.amm.execute_trade(wallet, trade_type, amount)
                delay = random.uniform(self.config.ai_parameters.min_delay, self.config.ai_parameters.max_delay)
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Market making iteration failed: {e}")
                await asyncio.sleep(5)
        logger.info("Hybrid market making completed")

    async def run_volume_boosting(self, token: str, use_random_amount: bool, amount: float, min_amount: float, max_amount: float, frequency: int, duration: int):
        logger.info(f"Starting volume boosting for {token} for {duration} minutes")
        self.is_running = True
        end_time = time.time() + duration * 60
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available")
            return
        while self.is_running and time.time() < end_time:
            for wallet in wallets:
                if not self.is_running or time.time() >= end_time:
                    break
                try:
                    trade_amount = random.uniform(min_amount, max_amount) if use_random_amount else amount
                    input_mint = "So11111111111111111111111111111111111111112"
                    output_mint = token
                    swap_amount = int(trade_amount * 1e9)
                    quote = await self.jupiter_client.get_quote(input_mint, output_mint, swap_amount, 100)
                    if 'error' not in quote:
                        swap_data = await self.jupiter_client.swap(quote, wallet)
                        if 'error' not in swap_data:
                            success = await self.jupiter_client.execute_swap_transaction(swap_data, wallet)
                            if success:
                                logger.info(f"Volume boost swap: {trade_amount} SOL -> {token}")
                    delay = 60.0 / frequency if frequency > 0 else 1.0
                    await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Volume boost iteration failed: {e}")
                    await asyncio.sleep(5)
        logger.info("Volume boosting completed")

    async def add_holders(self, count: int, min_amount: float, max_amount: float):
        logger.info(f"Starting holder simulation for {count} wallets")
        self.is_running = True
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available")
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
                quote = await self.jupiter_client.get_quote(input_mint, output_mint, swap_amount, 100)
                if 'error' not in quote:
                    swap_data = await self.jupiter_client.swap(quote, wallet)
                    if 'error' not in swap_data:
                        success = await self.jupiter_client.execute_swap_transaction(swap_data, wallet)
                        if success:
                            logger.info(f"Holder {i}/{len(selected_wallets)} added: {amount:.4f} SOL")
                await asyncio.sleep(random.uniform(self.config.ai_parameters.min_delay, self.config.ai_parameters.max_delay))
            except Exception as e:
                logger.error(f"Holder simulation failed for wallet {i}: {e}")
        logger.info("Holder simulation completed")

    async def withdraw_balances(self, token_mint: Optional[str], amount: Optional[float], decimals: int):
        logger.info(f"Starting balance withdrawal to main wallet: {self.config.main_wallet_pubkey}")
        wallets = await load_wallets(self.config)
        if not wallets:
            logger.error("No wallets available")
            return
        to_pubkey = Pubkey.from_string(self.config.main_wallet_pubkey)
        async with AsyncClient(self.config.rpc_url) as client:
            for i, wallet in enumerate(wallets, 1):
                try:
                    if token_mint:
                        token = Token(client, Pubkey.from_string(token_mint), TOKEN_PROGRAM_ID, wallet)
                        from_ata = await token.get_or_create_associated_account_info(wallet.pubkey())
                        to_ata = await token.get_or_create_associated_account_info(to_pubkey)
                        balance_resp = await client.get_token_account_balance(from_ata.address)
                        if not balance_resp.value:
                            logger.warning(f"Wallet {i} has no token balance")
                            continue
                        balance = balance_resp.value.amount
                        amount_to_transfer = int(amount * (10 ** decimals)) if amount else balance
                        if amount_to_transfer <= 0:
                            logger.warning(f"Wallet {i} has insufficient token balance")
                            continue
                        success = await transfer_spl_token(wallet, to_pubkey, token_mint, amount_to_transfer, decimals, self.config.rpc_url)
                        if success:
                            logger.info(f"Withdrew {amount_to_transfer / (10 ** decimals)} tokens from wallet {i}")
                    else:
                        balance_resp = await client.get_balance(wallet.pubkey())
                        if not balance_resp.value:
                            logger.warning(f"Wallet {i} has no SOL balance")
                            continue
                        balance = balance_resp.value
                        amount_to_transfer = int(amount * 1e9) if amount else max(0, balance - 1000000)
                        if amount_to_transfer <= 0:
                            logger.warning(f"Wallet {i} has insufficient SOL")
                            continue
                        success = await transfer_sol(wallet, to_pubkey, amount_to_transfer, self.config.rpc_url)
                        if success:
                            logger.info(f"Withdrew {amount_to_transfer / 1e9} SOL from wallet {i}")
                    await asyncio.sleep(random.uniform(self.config.ai_parameters.min_delay, self.config.ai_parameters.max_delay))
                except Exception as e:
                    logger.error(f"Withdrawal failed for wallet {i}: {e}")
        logger.info("Withdrawal completed")

    async def run_all_auto(self):
        logger.info("Starting all-auto strategy: Holders -> Volume -> Market Making")
        self.is_running = True

        # Validate configurations
        if not self.config.token_address:
            logger.error("Token address not set")
            raise ValueError("Set STEALTH_TOKEN_ADDRESS in .env")
        if self.config.holder_config.count <= 0 or self.config.holder_config.min_amount <= 0 or \
                self.config.holder_config.max_amount <= 0 or self.config.holder_config.min_amount > self.config.holder_config.max_amount:
            logger.error("Invalid holder configuration")
            raise ValueError("Set valid STEALTH_HOLDER_COUNT, STEALTH_HOLDER_MIN_AMOUNT, STEALTH_HOLDER_MAX_AMOUNT in .env")
        if not self.config.volume_boost_config.token or (self.config.volume_boost_config.use_random_amount and
                                                        (self.config.volume_boost_config.min_amount <= 0 or
                                                         self.config.volume_boost_config.max_amount <= 0 or
                                                         self.config.volume_boost_config.min_amount > self.config.volume_boost_config.max_amount)) or \
                (not self.config.volume_boost_config.use_random_amount and self.config.volume_boost_config.amount <= 0) or \
                self.config.volume_boost_config.frequency <= 0 or self.config.volume_boost_config.duration <= 0:
            logger.error("Invalid volume boost configuration")
            raise ValueError("Set valid volume boost parameters in .env")
        if self.config.hybrid_mm_config.duration_hours <= 0:
            logger.error("Invalid market making duration")
            raise ValueError("Set valid STEALTH_HYBRID_MM_DURATION_HOURS in .env")

        # Step 1: Holder Simulation
        try:
            logger.info("Executing Holder Simulation...")
            await self.add_holders(
                self.config.holder_config.count,
                self.config.holder_config.min_amount,
                self.config.holder_config.max_amount
            )
            logger.info("Holder Simulation completed")
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
            logger.info("Volume Boosting completed")
        except Exception as e:
            logger.error(f"Volume Boosting failed: {e}")
            self.is_running = False
            return

        # Step 3: Hybrid Market Making
        try:
            logger.info("Executing Hybrid Market Making...")
            await self.run_hybrid_market_making(self.config.hybrid_mm_config.duration_hours)
            logger.info("Hybrid Market Making completed")
        except Exception as e:
            logger.error(f"Hybrid Market Making failed: {e}")
            self.is_running = False
            return

        logger.info("All-auto strategy completed")

    def _adjust_amount_based_on_context(self, trade_type: TradeType, base_amount: float) -> float:
        if not self.amm.trade_history:
            return base_amount
        recent_trades = [t for t in self.amm.trade_history if time.time() - t['timestamp'] < 300]
        if not recent_trades:
            return base_amount
        buy_volume = sum(t['amount'] for t in recent_trades if t['type'] == 'buy')
        sell_volume = sum(t['amount'] for t in recent_trades if t['type'] == 'sell')
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return base_amount
        buy_ratio = buy_volume / total_volume
        if trade_type == TradeType.BUY and buy_ratio < 0.4:
            return base_amount * 1.2
        elif trade_type == TradeType.SELL and buy_ratio > 0.6:
            return base_amount * 1.2
        return base_amount

    def stop(self):
        self.is_running = False
        logger.info("Trading engine stopped")

# CLI Commands
@click.group()
@click.version_option(version='4.0.0')
def cli():
    pass

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def init(config_path: str):
    config = ConfigManager.load_config(config_path)
    ConfigManager.save_config(config)
    logger.info(f"Configuration initialized at {config_path}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--count', default=10, help='Number of wallets to generate')
def generate_wallets_cmd(config_path: str, count: int):
    config = ConfigManager.load_config(config_path)
    asyncio.run(generate_wallets(count, config))
    logger.info(f"Generated {count} wallets")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--duration', default=24, help='Duration in hours')
def hybrid_mm(config_path: str, duration: int):
    config = ConfigManager.load_config(config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_hybrid_market_making(duration))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Hybrid MM failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def hybrid_mm_auto(config_path: str):
    config = ConfigManager.load_config(config_path)
    if config.hybrid_mm_config.duration_hours <= 0:
        raise click.ClickException("Set a positive STEALTH_HYBRID_MM_DURATION_HOURS in .env")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_hybrid_market_making(config.hybrid_mm_config.duration_hours))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Hybrid MM failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--token', required=True, help='Token mint address')
@click.option('--use-random-amount', is_flag=True, help='Use random amounts')
@click.option('--amount', type=float, default=0.1, help='Fixed amount in SOL')
@click.option('--min-amount', type=float, default=0.01, help='Min random amount in SOL')
@click.option('--max-amount', type=float, default=0.1, help='Max random amount in SOL')
@click.option('--frequency', type=int, default=10, help='Swaps per minute')
@click.option('--duration', type=int, default=60, help='Duration in minutes')
def boost_volume(config_path: str, token: str, use_random_amount: bool, amount: float, min_amount: float, max_amount: float, frequency: int, duration: int):
    config = ConfigManager.load_config(config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_volume_boosting(token, use_random_amount, amount, min_amount, max_amount, frequency, duration))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Volume boosting failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def boost_volume_auto(config_path: str):
    config = ConfigManager.load_config(config_path)
    if not config.volume_boost_config.token or (config.volume_boost_config.use_random_amount and
                                               (config.volume_boost_config.min_amount <= 0 or
                                                config.volume_boost_config.max_amount <= 0 or
                                                config.volume_boost_config.min_amount > config.volume_boost_config.max_amount)) or \
            (not config.volume_boost_config.use_random_amount and config.volume_boost_config.amount <= 0) or \
            config.volume_boost_config.frequency <= 0 or config.volume_boost_config.duration <= 0:
        raise click.ClickException("Set valid volume boost parameters in .env")
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
        engine.stop()
    except Exception as e:
        logger.error(f"Volume boosting failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--input-mint', required=True, help='Input token mint')
@click.option('--output-mint', required=True, help='Output token mint')
@click.option('--amount', type=float, required=True, help='Amount to swap')
@click.option('--slippage', type=float, default=1.0, help='Slippage percentage')
def swap(config_path: str, input_mint: str, output_mint: str, amount: float, slippage: float):
    config = ConfigManager.load_config(config_path)
    async def execute_swap():
        async with aiohttp.ClientSession() as session:
            jupiter = JupiterLiteClient(session, config)
            wallets = await load_wallets(config)
            if not wallets:
                logger.error("No wallets available")
                return
            wallet = wallets[0]
            quote = await jupiter.get_quote(input_mint, output_mint, int(amount * 1e9), int(slippage * 100))
            if 'error' in quote:
                logger.error(f"Quote failed: {quote['error']}")
                return
            swap_data = await jupiter.swap(quote, wallet)
            if 'error' in swap_data:
                logger.error(f"Swap failed: {swap_data['error']}")
                return
            success = await jupiter.execute_swap_transaction(swap_data, wallet)
            logger.info("Swap executed successfully!" if success else "Swap execution failed")
    asyncio.run(execute_swap())

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--count', type=int, default=50, help='Number of wallets for holders')
@click.option('--min-amount', type=float, default=0.001, help='Min trade amount in SOL')
@click.option('--max-amount', type=float, default=0.05, help='Max trade amount in SOL')
def add_holders(config_path: str, count: int, min_amount: float, max_amount: float):
    config = ConfigManager.load_config(config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.add_holders(count, min_amount, max_amount))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Holder simulation failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def add_holders_auto(config_path: str):
    config = ConfigManager.load_config(config_path)
    if config.holder_config.count <= 0 or config.holder_config.min_amount <= 0 or \
            config.holder_config.max_amount <= 0 or config.holder_config.min_amount > config.holder_config.max_amount:
        raise click.ClickException("Set valid STEALTH_HOLDER_COUNT, STEALTH_HOLDER_MIN_AMOUNT, STEALTH_HOLDER_MAX_AMOUNT in .env")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.add_holders(
            config.holder_config.count,
            config.holder_config.min_amount,
            config.holder_config.max_amount
        ))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Holder simulation failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def all_auto(config_path: str):
    config = ConfigManager.load_config(config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_all_auto())
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"All-auto strategy failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
@click.option('--token-mint', default=None, help='Token mint address (omit for SOL)')
@click.option('--amount', type=float, default=None, help='Amount to withdraw')
@click.option('--decimals', type=int, default=9, help='Token decimals')
def withdraw(config_path: str, token_mint: Optional[str], amount: Optional[float], decimals: int):
    config = ConfigManager.load_config(config_path)
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.withdraw_balances(token_mint, amount, decimals))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Withdrawal failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def withdraw_auto(config_path: str):
    config = ConfigManager.load_config(config_path)
    if config.withdraw_config.amount is not None and config.withdraw_config.amount < 0:
        raise click.ClickException("Set a non-negative STEALTH_WITHDRAW_AMOUNT in .env")
    engine = StealthVolumeEngine(config)
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.withdraw_balances(
            config.withdraw_config.token_mint,
            config.withdraw_config.amount,
            config.withdraw_config.decimals
        ))
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.error(f"Withdrawal failed: {e}")

@cli.command()
@click.option('--config-path', default='config.yaml', help='Path to config file')
def status(config_path: str):
    config = ConfigManager.load_config(config_path)
    logger.info("=" * 50)
    logger.info("StealthVolume Status")
    logger.info("=" * 50)
    logger.info(f"Version: 4.0.0")
    logger.info(f"RPC URL: {config.rpc_url}")
    logger.info(f"Token Address: {config.token_address}")
    logger.info(f"DEXes: {', '.join(config.dexes)}")
    logger.info(f"Main Wallet: {config.main_wallet_pubkey}")
    logger.info(f"Wallets Loaded: {len(config.wallets)}")
    logger.info("Volume Boost Config:")
    logger.info(f"  Token: {config.volume_boost_config.token}")
    logger.info(f"  Use Random Amount: {config.volume_boost_config.use_random_amount}")
    logger.info(f"  Fixed Amount: {config.volume_boost_config.amount} SOL")
    logger.info(f"  Random Amount Range: {config.volume_boost_config.min_amount} - {config.volume_boost_config.max_amount} SOL")
    logger.info(f"  Frequency: {config.volume_boost_config.frequency} swaps/min")
    logger.info(f"  Duration: {config.volume_boost_config.duration} minutes")
    logger.info("Holder Config:")
    logger.info(f"  Count: {config.holder_config.count}")
    logger.info(f"  Amount Range: {config.holder_config.min_amount} - {config.holder_config.max_amount} SOL")
    logger.info("Hybrid MM Config:")
    logger.info(f"  Duration: {config.hybrid_mm_config.duration_hours} hours")
    logger.info(f"  Buy/Sell Ratio: {config.ai_parameters.buy_sell_ratio}")

if __name__ == '__main__':
    cli()

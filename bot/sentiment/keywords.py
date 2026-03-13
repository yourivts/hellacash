"""Dynamic keyword map for crypto asset sentiment matching."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Dict, List

logger = logging.getLogger(__name__)

# Hand-curated keywords for top assets (ticker → search terms)
_STATIC: Dict[str, List[str]] = {
    "BTC": ["bitcoin", "btc", "$btc"],
    "ETH": ["ethereum", "eth", "$eth", "vitalik"],
    "XRP": ["ripple", "xrp", "$xrp"],
    "SOL": ["solana", "sol", "$sol"],
    "ADA": ["cardano", "ada", "$ada"],
    "DOT": ["polkadot", "dot", "$dot"],
    "LINK": ["chainlink", "link", "$link"],
    "AVAX": ["avalanche", "avax", "$avax"],
    "MATIC": ["polygon", "matic", "$matic"],
    "DOGE": ["dogecoin", "doge", "$doge"],
    "SHIB": ["shiba", "shib", "$shib"],
    "UNI": ["uniswap", "uni", "$uni"],
    "AAVE": ["aave", "$aave"],
    "LTC": ["litecoin", "ltc", "$ltc"],
    "ATOM": ["cosmos", "atom", "$atom"],
    "FTM": ["fantom", "ftm", "$ftm"],
    "NEAR": ["near protocol", "near", "$near"],
    "ARB": ["arbitrum", "arb", "$arb"],
    "OP": ["optimism", "$op"],
    "APT": ["aptos", "apt", "$apt"],
    "SUI": ["sui", "$sui"],
    "FIL": ["filecoin", "fil", "$fil"],
    "INJ": ["injective", "inj", "$inj"],
    "ALGO": ["algorand", "algo", "$algo"],
    "VET": ["vechain", "vet", "$vet"],
    "ICP": ["internet computer", "icp", "$icp"],
    "GRT": ["the graph", "grt", "$grt"],
    "SAND": ["sandbox", "sand", "$sand"],
    "MANA": ["decentraland", "mana", "$mana"],
    "AXS": ["axie", "axs", "$axs"],
    "CRV": ["curve", "crv", "$crv"],
    "MKR": ["maker", "mkr", "$mkr"],
    "SNX": ["synthetix", "snx", "$snx"],
    "COMP": ["compound", "comp", "$comp"],
    "LDO": ["lido", "ldo", "$ldo"],
    "PEPE": ["pepe", "$pepe"],
    "WIF": ["dogwifhat", "wif", "$wif"],
    "BONK": ["bonk", "$bonk"],
    "FLOKI": ["floki", "$floki"],
    "TAO": ["bittensor", "tao", "$tao"],
    "RENDER": ["render", "rndr", "$rndr"],
    "FET": ["fetch.ai", "fet", "$fet"],
    "JASMY": ["jasmy", "$jasmy"],
    "XLM": ["stellar", "xlm", "$xlm"],
    "TRX": ["tron", "trx", "$trx"],
    "EOS": ["eos", "$eos"],
    "XTZ": ["tezos", "xtz", "$xtz"],
    "HBAR": ["hedera", "hbar", "$hbar"],
    "EGLD": ["multiversx", "elrond", "egld", "$egld"],
    "THETA": ["theta", "$theta"],
    "ENS": ["ens", "ethereum name service"],
    "DYDX": ["dydx", "$dydx"],
    "IMX": ["immutable", "imx", "$imx"],
    "QNT": ["quant", "qnt", "$qnt"],
    "SEI": ["sei", "$sei"],
    "TIA": ["celestia", "tia", "$tia"],
    "STX": ["stacks", "stx", "$stx"],
    "RUNE": ["thorchain", "rune", "$rune"],
}

# CoinGecko coin list cache (fetched once, maps symbol → coin name)
_coingecko_cache: Dict[str, List[str]] = {}
_coingecko_ts: float = 0
_COINGECKO_TTL = 86400  # 24 hours


def _load_coingecko() -> None:
    """Fetch CoinGecko coin list for symbol → name mapping."""
    global _coingecko_cache, _coingecko_ts
    now = time.time()
    if _coingecko_cache and (now - _coingecko_ts) < _COINGECKO_TTL:
        return
    try:
        req = urllib.request.Request(
            "https://api.coingecko.com/api/v3/coins/list",
            headers={"User-Agent": "hellacash/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            coins = json.loads(resp.read().decode())
        mapping: Dict[str, List[str]] = {}
        for coin in coins:
            sym = coin.get("symbol", "").upper()
            name = coin.get("name", "").lower()
            if sym and name:
                if sym not in mapping:
                    mapping[sym] = []
                if name not in mapping[sym]:
                    mapping[sym].append(name)
        _coingecko_cache = mapping
        _coingecko_ts = now
        logger.info("Loaded %d symbols from CoinGecko coin list", len(mapping))
    except Exception as e:
        logger.warning("CoinGecko coin list fetch failed: %s", e)


def get_keywords(asset: str) -> List[str]:
    """Return search keywords for an asset ticker. Tries static map, then CoinGecko, then falls back to ticker itself."""
    ticker = asset.upper()

    # Static map first
    if ticker in _STATIC:
        return _STATIC[ticker]

    # Try CoinGecko
    _load_coingecko()
    if ticker in _coingecko_cache:
        names = _coingecko_cache[ticker]
        keywords = [ticker.lower(), f"${ticker.lower()}"]
        for name in names:
            if name not in keywords and len(name) > 2:
                keywords.append(name)
        return keywords

    # Fallback: just match the ticker
    return [ticker.lower(), f"${ticker.lower()}"]

"""Bin-Sight: write a disposal record to Solana devnet as a memo transaction.

Uses `solders` to build and sign the transaction and plain JSON-RPC over HTTP to send it,
so it doesn't depend on the (frequently changing) solana-py client.

Set up / check your wallet:  python solana_log.py
"""
import base64
import json
import os
from pathlib import Path

import httpx
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
KEYPAIR_PATH = Path(os.getenv("SOLANA_KEYPAIR", "devnet-keypair.json"))
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")


class SolanaError(Exception):
    pass


def rpc(method: str, params: list):
    """Call a Solana JSON-RPC method and return its `result`."""
    resp = httpx.post(
        RPC_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise SolanaError(data["error"].get("message", str(data["error"])))
    return data["result"]


def load_or_create_keypair() -> Keypair:
    """The backend's own devnet wallet. The file is a private key: never commit it.
    On a host (Render etc.) set SOLANA_KEYPAIR_JSON to the file's contents instead."""
    env_json = os.getenv("SOLANA_KEYPAIR_JSON", "").strip()
    if env_json:
        return Keypair.from_bytes(bytes(json.loads(env_json)))
    if KEYPAIR_PATH.exists():
        return Keypair.from_bytes(bytes(json.loads(KEYPAIR_PATH.read_text())))
    kp = Keypair()
    KEYPAIR_PATH.write_text(json.dumps(list(bytes(kp))))
    return kp


def get_balance_sol(pubkey: Pubkey) -> float:
    return rpc("getBalance", [str(pubkey), {"commitment": "confirmed"}])["value"] / 1_000_000_000


def explorer_url(signature: str) -> str:
    return f"https://explorer.solana.com/tx/{signature}?cluster=devnet"


def log_disposal(record: dict) -> dict:
    """Send `record` as a compact JSON memo. Returns the signature and Explorer link."""
    kp = load_or_create_keypair()

    memo = json.dumps(record, separators=(",", ":")).encode()
    if len(memo) > 500:
        raise ValueError("Memo too long; keep the record small")

    instruction = Instruction(
        MEMO_PROGRAM_ID,
        memo,
        [AccountMeta(kp.pubkey(), is_signer=True, is_writable=True)],
    )
    blockhash = Hash.from_string(
        rpc("getLatestBlockhash", [{"commitment": "confirmed"}])["value"]["blockhash"]
    )
    tx = Transaction.new_signed_with_payer([instruction], kp.pubkey(), [kp], blockhash)
    signature = rpc(
        "sendTransaction",
        [base64.b64encode(bytes(tx)).decode(), {"encoding": "base64", "preflightCommitment": "confirmed"}],
    )
    return {"signature": signature, "explorer": explorer_url(signature)}


if __name__ == "__main__":
    kp = load_or_create_keypair()
    balance = get_balance_sol(kp.pubkey())
    print(f"Wallet address: {kp.pubkey()}")
    print(f"Devnet balance: {balance} SOL")
    if balance < 0.05:
        print("Low balance, requesting 1 devnet SOL airdrop...")
        try:
            rpc("requestAirdrop", [str(kp.pubkey()), 1_000_000_000])
            print("Requested. Re-run in ~30 seconds to check the balance.")
        except Exception as e:
            print(f"Airdrop failed ({e}).")
            print(f"Paste this address into https://faucet.solana.com instead: {kp.pubkey()}")

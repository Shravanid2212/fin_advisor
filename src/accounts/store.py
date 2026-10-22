"""User account authentication and secure profile/portfolio storage.

Adheres strictly to project rules:
1. Passwords hashed using hashlib.pbkdf2_hmac with random salt per user and high iteration count (200,000).
2. Constant-time comparison using hmac.compare_digest.
3. Minimum password length requirement (>= 8 characters).
4. Short lockout after repeated failed attempts (5 failed attempts locks for 15 minutes).
5. Never store or log plain passwords.
6. User profiles and portfolios stored in separate SQLite database (data/users.db), distinct from read-only market.duckdb.
7. Clear interface allowing future database swap.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "data/users.db"
HASH_ALGORITHM = "sha256"
ITERATIONS = 200_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 8
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 900  # 15 minutes


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open and initialize a connection to the users SQLite database."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Initialize user account and profile data tables."""
    con = get_connection(db_path)
    try:
        with con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS user_data (
                    username TEXT PRIMARY KEY,
                    profile_data TEXT,
                    portfolio_data TEXT,
                    settings_data TEXT,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
                """
            )
    finally:
        con.close()


def _hash_password(password: str, salt_hex: str) -> str:
    """Compute PBKDF2-HMAC-SHA256 hash of password with given salt."""
    salt_bytes = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode("utf-8"),
        salt_bytes,
        ITERATIONS,
    )
    return dk.hex()


def signup(
    username: str,
    password: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[bool, str]:
    """Register a new user account.

    Args:
        username: Desired username or email address.
        password: Plain text password (checked against minimum length and immediately hashed).
        db_path: Target SQLite database path.

    Returns:
        (success, message)
    """
    init_db(db_path)
    clean_user = username.strip().lower()
    if not clean_user or len(clean_user) < 3:
        return False, "Username must be at least 3 characters long."

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."

    salt_hex = os.urandom(SALT_BYTES).hex()
    pwd_hash = _hash_password(password, salt_hex)
    now = time.time()

    con = get_connection(db_path)
    try:
        with con:
            cur = con.execute("SELECT username FROM users WHERE username = ?", (clean_user,))
            if cur.fetchone() is not None:
                return False, f"Username '{clean_user}' is already registered."

            con.execute(
                """
                INSERT INTO users (username, password_hash, salt, failed_attempts, locked_until, created_at)
                VALUES (?, ?, ?, 0, 0, ?)
                """,
                (clean_user, pwd_hash, salt_hex, now),
            )
            con.execute(
                """
                INSERT INTO user_data (username, profile_data, portfolio_data, settings_data, updated_at)
                VALUES (?, NULL, NULL, NULL, ?)
                """,
                (clean_user, now),
            )
        return True, "Account created successfully."
    except Exception as exc:
        return False, f"Failed to create account: {exc}"
    finally:
        con.close()


def authenticate(
    username: str,
    password: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Authenticate a user with constant-time password comparison and lockout enforcement.

    Args:
        username: Account username.
        password: Plain text password to verify.
        db_path: Target SQLite database path.

    Returns:
        (success, message, user_data_dict)
    """
    init_db(db_path)
    clean_user = username.strip().lower()
    if not clean_user:
        return False, "Username cannot be empty.", None

    now = time.time()
    con = get_connection(db_path)
    try:
        cur = con.execute("SELECT * FROM users WHERE username = ?", (clean_user,))
        user_row = cur.fetchone()
        if user_row is None:
            # Perform dummy hash calculation to defend against timing attacks
            _hash_password("dummy_password_timing_defense", "00" * SALT_BYTES)
            return False, "Invalid username or password.", None

        # Check account lockout
        locked_until = float(user_row["locked_until"])
        if locked_until > now:
            remaining_mins = max(1, int((locked_until - now) / 60))
            return (
                False,
                f"Account is temporarily locked due to repeated failed login attempts. Try again in {remaining_mins} minute(s).",
                None,
            )

        # Verify password hash using constant-time comparison
        salt_hex = user_row["salt"]
        stored_hash = user_row["password_hash"]
        calc_hash = _hash_password(password, salt_hex)

        is_match = hmac.compare_digest(stored_hash, calc_hash)

        with con:
            if not is_match:
                new_fails = int(user_row["failed_attempts"]) + 1
                if new_fails >= MAX_FAILED_ATTEMPTS:
                    lock_time = now + LOCKOUT_DURATION_SECONDS
                    con.execute(
                        "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
                        (new_fails, lock_time, clean_user),
                    )
                    return (
                        False,
                        f"Account locked: {MAX_FAILED_ATTEMPTS} consecutive failed attempts. Locked for 15 minutes.",
                        None,
                    )
                else:
                    con.execute(
                        "UPDATE users SET failed_attempts = ? WHERE username = ?",
                        (new_fails, clean_user),
                    )
                    remaining_attempts = MAX_FAILED_ATTEMPTS - new_fails
                    return (
                        False,
                        f"Invalid username or password. ({remaining_attempts} attempt(s) remaining before temporary lockout).",
                        None,
                    )

            # Success: reset failed attempts and lockout
            con.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE username = ?",
                (clean_user,),
            )

        # Load saved user data
        data_dict = load_user_data(clean_user, db_path=db_path)
        return True, "Login successful.", data_dict
    finally:
        con.close()


def change_password(
    username: str,
    old_password: str,
    new_password: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[bool, str]:
    """Update a user's password securely."""
    init_db(db_path)
    clean_user = username.strip().lower()

    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"New password must be at least {MIN_PASSWORD_LENGTH} characters long."

    success, msg, _ = authenticate(clean_user, old_password, db_path=db_path)
    if not success:
        return False, f"Verification failed: {msg}"

    new_salt = os.urandom(SALT_BYTES).hex()
    new_hash = _hash_password(new_password, new_salt)

    con = get_connection(db_path)
    try:
        with con:
            con.execute(
                "UPDATE users SET password_hash = ?, salt = ?, failed_attempts = 0, locked_until = 0 WHERE username = ?",
                (new_hash, new_salt, clean_user),
            )
        return True, "Password updated successfully."
    finally:
        con.close()


def save_user_data(
    username: str,
    profile_data: dict[str, Any] | None = None,
    portfolio_data: dict[str, Any] | None = None,
    settings_data: dict[str, Any] | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Save or update user's profile, portfolio holdings, or settings.

    Pass None for any section that should remain unchanged.
    """
    init_db(db_path)
    clean_user = username.strip().lower()
    now = time.time()

    con = get_connection(db_path)
    try:
        cur = con.execute("SELECT * FROM user_data WHERE username = ?", (clean_user,))
        existing = cur.fetchone()

        cur_prof = existing["profile_data"] if existing else None
        cur_port = existing["portfolio_data"] if existing else None
        cur_sett = existing["settings_data"] if existing else None

        new_prof = json.dumps(profile_data) if profile_data is not None else cur_prof
        new_port = json.dumps(portfolio_data) if portfolio_data is not None else cur_port
        new_sett = json.dumps(settings_data) if settings_data is not None else cur_sett

        with con:
            con.execute(
                """
                INSERT INTO user_data (username, profile_data, portfolio_data, settings_data, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    profile_data = excluded.profile_data,
                    portfolio_data = excluded.portfolio_data,
                    settings_data = excluded.settings_data,
                    updated_at = excluded.updated_at
                """,
                (clean_user, new_prof, new_port, new_sett, now),
            )
        return True
    except Exception:
        return False
    finally:
        con.close()


def load_user_data(
    username: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Retrieve saved profile, portfolio, and settings for a user."""
    init_db(db_path)
    clean_user = username.strip().lower()
    con = get_connection(db_path)
    try:
        cur = con.execute("SELECT * FROM user_data WHERE username = ?", (clean_user,))
        row = cur.fetchone()
        if row is None:
            return {"profile_data": None, "portfolio_data": None, "settings_data": None}

        profile_data = json.loads(row["profile_data"]) if row["profile_data"] else None
        portfolio_data = json.loads(row["portfolio_data"]) if row["portfolio_data"] else None
        settings_data = json.loads(row["settings_data"]) if row["settings_data"] else None

        return {
            "profile_data": profile_data,
            "portfolio_data": portfolio_data,
            "settings_data": settings_data,
            "updated_at": row["updated_at"],
        }
    except Exception:
        return {"profile_data": None, "portfolio_data": None, "settings_data": None}
    finally:
        con.close()


def reset_portfolio_data(
    username: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Clear saved portfolio data for a user."""
    init_db(db_path)
    clean_user = username.strip().lower()
    now = time.time()
    con = get_connection(db_path)
    try:
        with con:
            con.execute(
                "UPDATE user_data SET portfolio_data = NULL, updated_at = ? WHERE username = ?",
                (now, clean_user),
            )
        return True
    except Exception:
        return False
    finally:
        con.close()


def delete_account(
    username: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Permanently delete a user account and all associated profile, portfolio, and settings data."""
    init_db(db_path)
    clean_user = username.strip().lower()
    con = get_connection(db_path)
    try:
        with con:
            con.execute("DELETE FROM user_data WHERE username = ?", (clean_user,))
            con.execute("DELETE FROM users WHERE username = ?", (clean_user,))
        return True
    except Exception:
        return False
    finally:
        con.close()

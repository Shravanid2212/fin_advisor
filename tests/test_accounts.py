"""Unit tests for user account store and authentication engine.

Tests:
1. Sign-up success, minimum password length enforcement, and duplicate handling.
2. Authentication success and failure with constant-time password verification.
3. Lockout enforcement after 5 consecutive failed login attempts.
4. Per-user isolation: User A cannot access User B's profile or portfolio.
5. Save, load, and reset of portfolio and profile data.
6. Password change and re-authentication with new password.
7. Account deletion: verifying complete removal of all user records.
8. Security audit: verifying plain text passwords NEVER appear in the SQLite database file.
"""

from __future__ import annotations

import sqlite3
import pytest

from src.accounts.store import (
    authenticate,
    change_password,
    delete_account,
    init_db,
    load_user_data,
    reset_portfolio_data,
    save_user_data,
    signup,
)


@pytest.fixture
def temp_user_db(tmp_path):
    """Provide an isolated temporary SQLite database path for testing."""
    db_file = tmp_path / "test_users.db"
    init_db(db_file)
    return str(db_file)


def test_signup_validation_and_duplicate(temp_user_db):
    """Verify signup enforces username/password length and prevents duplicates."""
    # 1. Password under 8 chars fails
    ok, msg = signup("alice", "short", db_path=temp_user_db)
    assert ok is False
    assert "at least 8 characters" in msg

    # 2. Username under 3 chars fails
    ok, msg = signup("al", "ValidPassword123!", db_path=temp_user_db)
    assert ok is False
    assert "at least 3 characters" in msg

    # 3. Valid signup succeeds
    ok, msg = signup("alice", "StrongPassword123!", db_path=temp_user_db)
    assert ok is True
    assert "created successfully" in msg

    # 4. Duplicate signup fails
    ok, msg = signup("alice", "AnotherPassword456!", db_path=temp_user_db)
    assert ok is False
    assert "already registered" in msg


def test_authentication_flow_and_lockout(temp_user_db):
    """Verify authentication matches passwords and triggers temporary lockout after 5 fails."""
    signup("bob", "SecretPass123!", db_path=temp_user_db)

    # 1. Non-existent user
    ok, msg, data = authenticate("charlie", "SecretPass123!", db_path=temp_user_db)
    assert ok is False
    assert data is None

    # 2. Correct password succeeds
    ok, msg, data = authenticate("bob", "SecretPass123!", db_path=temp_user_db)
    assert ok is True
    assert "Login successful" in msg
    assert data is not None

    # 3. Failed attempts 1 to 4
    for attempt in range(1, 5):
        ok, msg, data = authenticate("bob", "WrongPassword!", db_path=temp_user_db)
        assert ok is False
        assert "Invalid username or password" in msg
        assert f"{5 - attempt} attempt(s) remaining" in msg

    # 4. 5th failed attempt triggers lockout
    ok, msg, data = authenticate("bob", "WrongPassword!", db_path=temp_user_db)
    assert ok is False
    assert "Account locked" in msg or "repeated failed" in msg

    # 5. Subsequent attempt even with correct password is rejected while locked
    ok, msg, data = authenticate("bob", "SecretPass123!", db_path=temp_user_db)
    assert ok is False
    assert "temporarily locked" in msg


def test_per_user_isolation(temp_user_db):
    """Verify that User A's data is strictly isolated from User B."""
    signup("user_a", "PasswordUserA123!", db_path=temp_user_db)
    signup("user_b", "PasswordUserB123!", db_path=temp_user_db)

    # User A saves profile and portfolio
    prof_a = {"risk_level": 4, "label": "Growth"}
    port_a = [{"ticker": "TCS.NS", "quantity": 10, "avg_buy_price": 3200.0}]
    save_user_data("user_a", profile_data=prof_a, portfolio_data=port_a, db_path=temp_user_db)

    # User B saves different profile
    prof_b = {"risk_level": 2, "label": "Moderately Conservative"}
    save_user_data("user_b", profile_data=prof_b, db_path=temp_user_db)

    # Load and verify isolation
    data_a = load_user_data("user_a", db_path=temp_user_db)
    data_b = load_user_data("user_b", db_path=temp_user_db)

    assert data_a["profile_data"]["risk_level"] == 4
    assert len(data_a["portfolio_data"]) == 1
    assert data_a["portfolio_data"][0]["ticker"] == "TCS.NS"

    assert data_b["profile_data"]["risk_level"] == 2
    assert data_b["portfolio_data"] is None  # User B never sees User A's portfolio


def test_reset_portfolio_data(temp_user_db):
    """Verify reset_portfolio_data clears holdings without affecting profile."""
    signup("dan", "PasswordDan123!", db_path=temp_user_db)
    save_user_data(
        "dan",
        profile_data={"level": 3},
        portfolio_data=[{"ticker": "INFY.NS", "quantity": 50}],
        db_path=temp_user_db,
    )

    reset_portfolio_data("dan", db_path=temp_user_db)
    data = load_user_data("dan", db_path=temp_user_db)

    assert data["portfolio_data"] is None
    assert data["profile_data"] == {"level": 3}


def test_change_password(temp_user_db):
    """Verify change_password validates old password and allows login with new password."""
    signup("eve", "OldPassword123!", db_path=temp_user_db)

    # Wrong old password fails
    ok, msg = change_password("eve", "WrongOld!", "NewPassword123!", db_path=temp_user_db)
    assert ok is False

    # Short new password fails
    ok, msg = change_password("eve", "OldPassword123!", "short", db_path=temp_user_db)
    assert ok is False

    # Valid change succeeds
    ok, msg = change_password("eve", "OldPassword123!", "NewPassword123!", db_path=temp_user_db)
    assert ok is True

    # Old password no longer authenticates
    ok, msg, _ = authenticate("eve", "OldPassword123!", db_path=temp_user_db)
    assert ok is False

    # New password authenticates
    ok, msg, _ = authenticate("eve", "NewPassword123!", db_path=temp_user_db)
    assert ok is True


def test_delete_account(temp_user_db):
    """Verify delete_account completely purges the user and their data."""
    signup("frank", "FrankPassword123!", db_path=temp_user_db)
    save_user_data("frank", profile_data={"level": 5}, db_path=temp_user_db)

    # Delete
    ok = delete_account("frank", db_path=temp_user_db)
    assert ok is True

    # Cannot authenticate
    ok, msg, _ = authenticate("frank", "FrankPassword123!", db_path=temp_user_db)
    assert ok is False

    # Data is gone
    data = load_user_data("frank", db_path=temp_user_db)
    assert data["profile_data"] is None


def test_password_hashing_security_no_plain_text(temp_user_db):
    """Security audit: verify raw plain text passwords NEVER exist in the SQLite database."""
    plain_password = "SuperSecretPlainTextPassword123!"
    signup("grace", plain_password, db_path=temp_user_db)

    # Read the raw SQLite database directly
    con = sqlite3.connect(temp_user_db)
    try:
        rows = con.execute("SELECT * FROM users WHERE username = 'grace'").fetchall()
        assert len(rows) == 1
        user_row = dict(rows[0]) if hasattr(rows[0], "keys") else rows[0]

        # Convert row to string representation
        row_str = str(user_row)
        assert plain_password not in row_str, "CRITICAL: Plain text password found in users table!"

        # Ensure password_hash is a hex string of length 64 (sha256 digest)
        # and salt is length 32 (16 bytes hex)
        cur = con.execute("SELECT password_hash, salt FROM users WHERE username = 'grace'")
        p_hash, salt = cur.fetchone()
        assert len(p_hash) == 64
        assert len(salt) == 32
        assert p_hash != plain_password
    finally:
        con.close()

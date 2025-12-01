"""
Database utilities for user authentication and data storage
"""

import sqlite3
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import bcrypt
import re

class DatabaseManager:
    """SQLite database manager for user authentication and data storage"""

    def __init__(self, db_path: str = "users.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    company TEXT,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    email_verified BOOLEAN DEFAULT 0,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            ''')

            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    theme TEXT DEFAULT 'light',
                    default_model TEXT DEFAULT 'lightgbm',
                    dashboard_layout TEXT,
                    notifications_enabled BOOLEAN DEFAULT 1,
                    auto_save BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Activity logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Password reset tokens table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at DATETIME NOT NULL,
                    used BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            conn.commit()

class PasswordManager:
    """Password hashing and validation utilities"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, str]:
        """Validate password strength requirements"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"

        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"

        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"

        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"

        return True, "Password meets strength requirements"

class UserManager:
    """User management operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_user(self, username: str, email: str, password: str,
                   full_name: str = None, company: str = None) -> Tuple[bool, str]:
        """Create a new user account"""
        try:
            # Validate input
            if not username or not email or not password:
                return False, "All fields are required"

            # Validate email format
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return False, "Invalid email format"

            # Validate password strength
            is_strong, msg = PasswordManager.validate_password_strength(password)
            if not is_strong:
                return False, msg

            # Hash password
            password_hash = PasswordManager.hash_password(password)

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Check if username or email already exists
                cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?",
                             (username, email))
                if cursor.fetchone():
                    return False, "Username or email already exists"

                # Create user
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, full_name, company)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, password_hash, full_name, company))

                user_id = cursor.lastrowid

                # Create default preferences
                cursor.execute('''
                    INSERT INTO user_preferences (user_id)
                    VALUES (?)
                ''', (user_id,))

                # Log activity
                self.log_activity(user_id, 'user_registration', f'User {username} registered')

                conn.commit()

                return True, f"User {username} created successfully"

        except Exception as e:
            return False, f"Error creating user: {str(e)}"

    def authenticate_user(self, username_or_email: str, password: str,
                         ip_address: str = None, user_agent: str = None) -> Tuple[bool, str, Optional[Dict]]:
        """Authenticate a user"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get user by username or email
                cursor.execute('''
                    SELECT id, username, email, password_hash, role, is_active,
                           failed_login_attempts, locked_until, full_name, company
                    FROM users
                    WHERE (username = ? OR email = ?) AND is_active = 1
                ''', (username_or_email, username_or_email))

                user = cursor.fetchone()
                if not user:
                    return False, "Invalid username or password", None

                user_id, username, email, password_hash, role, is_active, \
                failed_attempts, locked_until, full_name, company = user

                # Check if account is locked
                if locked_until and datetime.now() > datetime.fromisoformat(locked_until):
                    # Unlock account if lock time has passed
                    cursor.execute("UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?",
                                 (user_id,))
                    failed_attempts = 0
                    locked_until = None
                    conn.commit()

                if locked_until:
                    return False, f"Account locked until {locked_until}", None

                # Verify password
                if not PasswordManager.verify_password(password, password_hash):
                    # Increment failed attempts
                    failed_attempts += 1
                    lock_until = None

                    if failed_attempts >= 5:
                        # Lock account for 30 minutes
                        lock_until = (datetime.now() + timedelta(minutes=30)).isoformat()
                        cursor.execute('''
                            UPDATE users SET failed_login_attempts = ?, locked_until = ?
                            WHERE id = ?
                        ''', (failed_attempts, lock_until, user_id))
                        conn.commit()
                        return False, "Account locked due to too many failed attempts", None
                    else:
                        cursor.execute('''
                            UPDATE users SET failed_login_attempts = ?
                            WHERE id = ?
                        ''', (failed_attempts, user_id))
                        conn.commit()
                        return False, f"Invalid password. {5 - failed_attempts} attempts remaining", None

                # Successful login - reset failed attempts and update last login
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = 0, locked_until = NULL,
                                     last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))

                # Log successful login
                self.log_activity(user_id, 'login', f'User {username} logged in',
                                ip_address, user_agent)

                conn.commit()

                user_data = {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'role': role,
                    'full_name': full_name,
                    'company': company
                }

                return True, "Login successful", user_data

        except Exception as e:
            return False, f"Authentication error: {str(e)}", None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user information by ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, username, email, full_name, company, role,
                           created_at, last_login
                    FROM users WHERE id = ? AND is_active = 1
                ''', (user_id,))

                user = cursor.fetchone()
                if user:
                    return {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'full_name': user[3],
                        'company': user[4],
                        'role': user[5],
                        'created_at': user[6],
                        'last_login': user[7]
                    }
                return None
        except Exception:
            return None

    def update_user_profile(self, user_id: int, updates: Dict) -> Tuple[bool, str]:
        """Update user profile information"""
        try:
            allowed_fields = ['full_name', 'company', 'email']
            update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

            if not update_fields:
                return False, "No valid fields to update"

            # Check email uniqueness if email is being updated
            if 'email' in update_fields:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM users WHERE email = ? AND id != ?",
                                 (update_fields['email'], user_id))
                    if cursor.fetchone():
                        return False, "Email already in use"

            # Build update query
            set_clause = ', '.join([f"{field} = ?" for field in update_fields.keys()])
            values = list(update_fields.values()) + [user_id]

            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE users SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', values)

                if cursor.rowcount > 0:
                    self.log_activity(user_id, 'profile_update',
                                    f'Updated fields: {", ".join(update_fields.keys())}')
                    conn.commit()
                    return True, "Profile updated successfully"
                else:
                    return False, "User not found or no changes made"

        except Exception as e:
            return False, f"Error updating profile: {str(e)}"

    def change_password(self, user_id: int, current_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        try:
            # Validate new password strength
            is_strong, msg = PasswordManager.validate_password_strength(new_password)
            if not is_strong:
                return False, msg

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get current password hash
                cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
                result = cursor.fetchone()
                if not result:
                    return False, "User not found"

                # Verify current password
                if not PasswordManager.verify_password(current_password, result[0]):
                    return False, "Current password is incorrect"

                # Hash new password
                new_hash = PasswordManager.hash_password(new_password)

                # Update password
                cursor.execute('''
                    UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_hash, user_id))

                # Log password change
                self.log_activity(user_id, 'password_change', 'Password changed successfully')

                conn.commit()
                return True, "Password changed successfully"

        except Exception as e:
            return False, f"Error changing password: {str(e)}"

    def log_activity(self, user_id: int, action: str, details: str = None,
                    ip_address: str = None, user_agent: str = None):
        """Log user activity"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, action, details, ip_address, user_agent))
                conn.commit()
        except Exception:
            pass  # Don't fail the main operation due to logging errors

class SessionManager:
    """Session management for user authentication"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_session(self, user_id: int, ip_address: str = None,
                      user_agent: str = None) -> str:
        """Create a new session for user"""
        try:
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)  # 24 hour sessions

            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_sessions (user_id, session_token, ip_address,
                                             user_agent, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, session_token, ip_address, user_agent, expires_at))

                conn.commit()
                return session_token

        except Exception as e:
            raise Exception(f"Error creating session: {str(e)}")

    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user data"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT u.id, u.username, u.email, u.role, u.full_name, u.company,
                           s.expires_at
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.session_token = ? AND u.is_active = 1
                ''', (session_token,))

                result = cursor.fetchone()
                if result:
                    user_id, username, email, role, full_name, company, expires_at = result

                    # Check if session is expired
                    if datetime.now() > datetime.fromisoformat(expires_at):
                        # Clean up expired session
                        cursor.execute("DELETE FROM user_sessions WHERE session_token = ?",
                                     (session_token,))
                        conn.commit()
                        return None

                    return {
                        'id': user_id,
                        'username': username,
                        'email': email,
                        'role': role,
                        'full_name': full_name,
                        'company': company
                    }

                return None

        except Exception:
            return None

    def destroy_session(self, session_token: str):
        """Destroy a user session"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_sessions WHERE session_token = ?",
                             (session_token,))
                conn.commit()
        except Exception:
            pass

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP")
                conn.commit()
        except Exception:
            pass

# Global instances
db_manager = DatabaseManager()
user_manager = UserManager(db_manager)
session_manager = SessionManager(db_manager)

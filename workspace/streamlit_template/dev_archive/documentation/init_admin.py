#!/usr/bin/env python3
"""
Initialize database with default admin user
Run this script once to create the initial admin user
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import user_manager

def create_default_admin():
    """Create default admin user"""
    print("🔐 Creating default admin user...")

    # Default admin credentials
    admin_username = "admin"
    admin_email = "admin@retailforecast.com"
    admin_password = "Admin123!"

    success, message = user_manager.create_user(
        admin_username,
        admin_email,
        admin_password,
        "System Administrator",
        "Retail Forecasting Platform"
    )

    if success:
        print("✅ Default admin user created successfully!")
        print(f"   Username: {admin_username}")
        print(f"   Email: {admin_email}")
        print(f"   Password: {admin_password}")
        print("\n⚠️  IMPORTANT: Change the default password after first login!")
        print("   Go to Authentication page → Login → Update Profile → Change Password")
    else:
        print(f"❌ Failed to create admin user: {message}")

    # Update admin role
    with user_manager.db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role = 'admin' WHERE username = ?", (admin_username,))
        conn.commit()
        print("✅ Admin role assigned successfully!")

if __name__ == "__main__":
    create_default_admin()

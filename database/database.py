import sqlite3
import hashlib
from pathlib import Path

# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "fertilizer.db"

def get_connection():
    """Create and return a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """Convert password into a secure SHA-256 hash."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def initialize_database():
    """Create required tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT,
            state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    
    # Add missing columns to older users databases
    cursor.execute("PRAGMA table_info(users)")
    existing_user_cols = {row["name"] for row in cursor.fetchall()}

    if "phone" not in existing_user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN phone TEXT")

    if "state" not in existing_user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN state TEXT")


    # Recommendations & Soil Test History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            email TEXT,
            crop TEXT NOT NULL,
            soil_type TEXT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            soil_ph REAL,
            moisture REAL,
            organic_carbon REAL,
            electrical_conductivity REAL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            growth_stage TEXT,
            season TEXT,
            irrigation_type TEXT,
            region TEXT,
            plot_area REAL DEFAULT 1.0,
            area_unit TEXT DEFAULT 'Acre',
            budget REAL,
            recommended_fertilizer TEXT NOT NULL,
            confidence REAL,
            dosage_kg REAL,
            bags_count REAL,
            estimated_cost REAL,
            standard_cost REAL,
            savings_amount REAL,
            application_schedule TEXT,
            soil_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Farm Profiles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farm_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            farm_name TEXT NOT NULL,
            location TEXT,
            soil_type TEXT,
            primary_crop TEXT,
            total_acres REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crop_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_name TEXT NOT NULL,
            location TEXT,
            crop TEXT NOT NULL,
            quantity_quintals REAL NOT NULL,
            expected_price REAL NOT NULL,
            contact TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed sample demonstration crop listings if table is empty
    cursor.execute("SELECT COUNT(*) FROM crop_listings")
    if cursor.fetchone()[0] == 0:
        sample_crops = [
            ("Ramesh Singh", "Ludhiana, Punjab", "Wheat", 65.0, 2275.0, "98765-XXXX1"),
            ("Suresh Patel", "Rajkot, Gujarat", "Cotton", 40.0, 7120.0, "98765-XXXX2"),
            ("Anita Sharma", "Karnal, Haryana", "Rice", 80.0, 2300.0, "98765-XXXX3"),
            ("Balwan Yadav", "Bhopal, MP", "Soybean", 35.0, 4890.0, "98765-XXXX4")
        ]
        cursor.executemany("""
            INSERT INTO crop_listings (farmer_name, location, crop, quantity_quintals, expected_price, contact)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_crops)

    # Ensure all expected columns exist in recommendations for backward compatibility
    cursor.execute("PRAGMA table_info(recommendations)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    expected_columns = {
        "email": "TEXT",
        "soil_type": "TEXT",
        "moisture": "REAL",
        "organic_carbon": "REAL",
        "electrical_conductivity": "REAL",
        "temperature": "REAL",
        "humidity": "REAL",
        "rainfall": "REAL",
        "growth_stage": "TEXT",
        "season": "TEXT",
        "irrigation_type": "TEXT",
        "region": "TEXT",
        "plot_area": "REAL DEFAULT 1.0",
        "area_unit": "TEXT DEFAULT 'Acre'",
        "confidence": "REAL",
        "dosage_kg": "REAL",
        "bags_count": "REAL",
        "estimated_cost": "REAL",
        "standard_cost": "REAL",
        "savings_amount": "REAL",
        "application_schedule": "TEXT",
        "soil_status": "TEXT"
    }

    for col_name, col_type in expected_columns.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE recommendations ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()

def create_user(name, email, password, phone="", state=""):
    """Create a new user account."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        hashed_password = hash_password(password)
        cursor.execute("""
            INSERT INTO users (name, email, password, phone, state)
            VALUES (?, ?, ?, ?, ?)
        """, (name.strip(), email.strip().lower(), hashed_password, phone.strip(), state.strip()))

        user_id = cursor.lastrowid
        conn.commit()
        return True, {"id": user_id, "name": name, "email": email.strip().lower()}

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def authenticate_user(email, password):
    """Verify login credentials."""
    conn = get_connection()
    cursor = conn.cursor()

    hashed_password = hash_password(password)
    cursor.execute("""
        SELECT id, name, email, phone, state
        FROM users
        WHERE email = ? AND password = ?
    """, (email.strip().lower(), hashed_password))

    user = cursor.fetchone()
    conn.close()

    if user:
        return dict(user)
    return None

def save_recommendation_record(data: dict):
    """Save full recommendation history record."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO recommendations (
                user_id, email, crop, soil_type, nitrogen, phosphorus, potassium,
                soil_ph, moisture, organic_carbon, electrical_conductivity,
                temperature, humidity, rainfall, growth_stage, season,
                irrigation_type, region, plot_area, area_unit, budget,
                recommended_fertilizer, confidence, dosage_kg, bags_count,
                estimated_cost, standard_cost, savings_amount,
                application_schedule, soil_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("user_id"),
            data.get("email", "farmer@agrismart.local"),
            data.get("crop", "Wheat"),
            data.get("soil_type", "Loamy"),
            data.get("nitrogen", 0.0),
            data.get("phosphorus", 0.0),
            data.get("potassium", 0.0),
            data.get("soil_ph", 7.0),
            data.get("moisture", 30.0),
            data.get("organic_carbon", 0.5),
            data.get("electrical_conductivity", 1.0),
            data.get("temperature", 25.0),
            data.get("humidity", 60.0),
            data.get("rainfall", 1000.0),
            data.get("growth_stage", "Vegetative"),
            data.get("season", "Kharif"),
            data.get("irrigation_type", "Canal"),
            data.get("region", "North"),
            data.get("plot_area", 1.0),
            data.get("area_unit", "Acre"),
            data.get("budget", 2000.0),
            data.get("recommended_fertilizer", "Urea"),
            data.get("confidence", 95.0),
            data.get("dosage_kg", 50.0),
            data.get("bags_count", 1.0),
            data.get("estimated_cost", 300.0),
            data.get("standard_cost", 500.0),
            data.get("savings_amount", 200.0),
            data.get("application_schedule", ""),
            data.get("soil_status", "")
        ))

        rec_id = cursor.lastrowid
        conn.commit()
        return rec_id
    except Exception as e:
        print(f"Error saving recommendation: {e}")
        return None
    finally:
        conn.close()

def get_recommendations(user_id=None, email=None, limit=50):
    """Retrieve recommendation history records."""
    conn = get_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT * FROM recommendations
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
        """, (user_id, limit))
    elif email and email != "guest":
        cursor.execute("""
            SELECT * FROM recommendations
            WHERE email = ?
            ORDER BY id DESC LIMIT ?
        """, (email, limit))
    else:
        cursor.execute("""
            SELECT * FROM recommendations
            ORDER BY id DESC LIMIT ?
        """, (limit,))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_recommendation(rec_id, user_id=None):
    """Delete a single recommendation record."""
    conn = get_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("DELETE FROM recommendations WHERE id = ? AND user_id = ?", (rec_id, user_id))
    else:
        cursor.execute("DELETE FROM recommendations WHERE id = ?", (rec_id,))

    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_aggregated_stats():
    """Retrieve statistical summaries for analytics dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM recommendations")
    total_recs = cursor.fetchone()[0]

    cursor.execute("""
        SELECT recommended_fertilizer, COUNT(*) as cnt
        FROM recommendations
        GROUP BY recommended_fertilizer
        ORDER BY cnt DESC
    """)
    fert_dist = [dict(row) for row in cursor.fetchall()]

    cursor.execute("""
        SELECT crop, COUNT(*) as cnt
        FROM recommendations
        GROUP BY crop
        ORDER BY cnt DESC
        LIMIT 6
    """)
    crop_dist = [dict(row) for row in cursor.fetchall()]

    cursor.execute("""
        SELECT 
            AVG(nitrogen) as avg_n,
            AVG(phosphorus) as avg_p,
            AVG(potassium) as avg_k,
            AVG(soil_ph) as avg_ph,
            AVG(savings_amount) as avg_savings,
            SUM(savings_amount) as total_savings
        FROM recommendations
    """)
    avg_row = cursor.fetchone()
    conn.close()

    return {
        "total_recommendations": total_recs,
        "fertilizer_distribution": fert_dist,
        "crop_distribution": crop_dist,
        "averages": {
            "nitrogen": round(avg_row["avg_n"] or 0, 1) if avg_row else 0,
            "phosphorus": round(avg_row["avg_p"] or 0, 1) if avg_row else 0,
            "potassium": round(avg_row["avg_k"] or 0, 1) if avg_row else 0,
            "soil_ph": round(avg_row["avg_ph"] or 0, 2) if avg_row else 7.0,
            "avg_savings": round(avg_row["avg_savings"] or 0, 2) if avg_row else 0,
            "total_savings": round(avg_row["total_savings"] or 0, 2) if avg_row else 0
        }
    }

def add_crop_listing(farmer_name, location, crop, quantity_quintals, expected_price, contact=""):
    """Inserts a new sample crop listing into the marketplace demonstration database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO crop_listings (farmer_name, location, crop, quantity_quintals, expected_price, contact)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(farmer_name or "Farmer").strip(),
        str(location or "Local Mandi").strip(),
        str(crop or "Wheat").strip(),
        float(quantity_quintals or 1.0),
        float(expected_price or 1000.0),
        str(contact or "").strip()
    ))
    listing_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return listing_id

def get_crop_listings(limit=25):
    """Retrieves all active crop listings ordered by newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, farmer_name, location, crop, quantity_quintals, expected_price, contact, created_at
        FROM crop_listings
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# Run initialization on import
initialize_database()
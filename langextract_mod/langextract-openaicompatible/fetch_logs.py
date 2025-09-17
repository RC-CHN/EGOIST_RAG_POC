"""
Script to fetch chat logs from the PostgreSQL database and save them to a local
JSON file for later processing.
"""
import json
import psycopg2

# --- Database Configuration ---
DB_CONFIG = {
    "host": "192.168.44.9",
    "port": "5432",
    "dbname": "chatlog",
    "user": "user",
    "password": "password",
}

# --- Query Parameters ---
TARGET_GROUP_ID = '975206796'
MESSAGE_LIMIT = 5000
OUTPUT_FILE = 'chatlog.json'

def fetch_and_save_logs():
    """
    Connects to the database, fetches specified chat messages, and saves them
    to a JSON file.
    """
    print(f"Connecting to database '{DB_CONFIG['dbname']}'...")
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print(f"Executing query for group_id={TARGET_GROUP_ID}, limit={MESSAGE_LIMIT}...")
        
        # Execute the query with specific group_id and limit
        cur.execute(
            """
            SELECT timestamp, raw_data, message_str
            FROM chat_messages
            WHERE group_id = %s
              AND message_str IS NOT NULL 
              AND message_str != '' 
            ORDER BY timestamp DESC 
            LIMIT %s
            """,
            (TARGET_GROUP_ID, MESSAGE_LIMIT)
        )
        
        rows = cur.fetchall()
        print(f"✅ Fetched {len(rows)} messages from the database.")
        
        # Process rows into a list of dictionaries
        chat_logs = []
        for row in rows:
            timestamp, raw_data, message_str = row
            # psycopg2 automatically decodes json/jsonb columns into dicts.
            # No need for json.loads().
            sender_data = raw_data.get("sender", {})
            chat_logs.append({
                "timestamp": timestamp,
                "user_id": sender_data.get("user_id"),
                "nickname": sender_data.get("nickname"),
                "message": message_str,
            })
            
        # Save to a local JSON file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_logs, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Successfully saved {len(chat_logs)} messages to '{OUTPUT_FILE}'.")

    except psycopg2.OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        print("   Please ensure the database is accessible and credentials are correct.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    fetch_and_save_logs()
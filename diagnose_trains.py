"""Find train that exists in both tickets + trains table, with booking date info."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import psycopg2
from urllib.parse import urlparse, unquote

DB_URL = os.getenv("DB_URL") or (
    "postgresql://postgres.trssafmvdbeagsdnivcl:"
    "digitalshield%4023070802"
    "@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
)
p = urlparse(DB_URL)
conn = psycopg2.connect(
    host=p.hostname, port=p.port or 5432,
    dbname=p.path.lstrip("/"),
    user=p.username, password=unquote(p.password or ""),
    sslmode="require", options="-c search_path=public"
)
cur = conn.cursor()

# Top trains in tickets_1 by booking count
cur.execute("""
    SELECT train_number::text, COUNT(*) as cnt
    FROM tickets_1
    GROUP BY train_number
    ORDER BY cnt DESC
    LIMIT 20
""")
top_trains = [r[0] for r in cur.fetchall()]
print("Top trains in tickets_1:", top_trains[:5])

# Find which ones exist in trains table
placeholders = ",".join([f"'{t}'" for t in top_trains])
cur.execute(f"""
    SELECT train_number, train_name, platform_number, departure_days, departure_time
    FROM trains
    WHERE train_number IN ({placeholders})
    LIMIT 10
""")
matches = cur.fetchall()
print(f"\n=== Trains in BOTH tickets_1 AND trains table ({len(matches)} found) ===")
for m in matches:
    print(f"  train={m[0]} | name={m[1]} | platform={m[2]} | days={m[3]} | time={m[4]}")

if matches:
    pick = matches[0]
    train_num = pick[0]
    platform  = pick[2]
    days_str  = pick[3]
    dep_time  = pick[4]

    # Get a real journey date for this train
    cur.execute(f"""
        SELECT jrny_date, COUNT(*) as cnt
        FROM tickets_1
        WHERE train_number::text = '{train_num}'
        GROUP BY jrny_date
        ORDER BY cnt DESC
        LIMIT 5
    """)
    dates = cur.fetchall()
    print(f"\n=== Journey dates for Train {train_num} ===")
    for d in dates:
        print(f"  date={d[0]} | bookings={d[1]}")

    print(f"\n=== RECOMMENDED TEST PAYLOAD ===")
    import datetime
    if dates:
        jrny_date = dates[0][0]
        # parse day of week from the date
        try:
            dt = datetime.datetime.strptime(jrny_date, "%Y-%m-%d")
            day_name = dt.strftime("%A")
        except:
            day_name = "Monday"
        
        print(f"  platform_number: {platform}")
        print(f"  date:            {jrny_date}")
        print(f"  day:             {day_name}")
        print(f"  time:            {str(dep_time)[:5] if dep_time else '10:00'}")
        print(f"  train_number:    {train_num}")
        print(f"  departure_days:  {days_str}")

conn.close()

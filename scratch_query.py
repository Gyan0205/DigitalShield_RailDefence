from backend.database import db_service
from sqlalchemy import text

with db_service.engine.connect() as conn:
    res = conn.execute(text("SELECT * FROM trains WHERE platform_number=3 AND departure_days LIKE '%Wed%'")).fetchall()
    for row in res:
        print(dict(row._mapping))

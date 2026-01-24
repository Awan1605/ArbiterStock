# python_ai/database.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from python_ai.config.settings import DATABASE_URL

# ================= DATABASE ENGINE =================
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)

# ================= SESSION =================
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# ================= DEPENDENCY FASTAPI =================
def get_db():
    """
    Yield database session untuk dependency injection FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================= TEST CONNECTION =================
def test_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print("❌ DB ERROR:", e)
        return False

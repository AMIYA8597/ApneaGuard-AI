import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Fallback to local docker defaults if not provided
DB_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/apneaguard")

from sqlalchemy.pool import StaticPool

if os.environ.get("TESTING") == "1":
    DB_URL = "sqlite:///:memory:"
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
else:
    engine = create_engine(DB_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

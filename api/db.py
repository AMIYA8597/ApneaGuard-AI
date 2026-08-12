import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

DB_URL = settings.DATABASE_URL

if os.environ.get("TESTING") == "1":
    DB_URL = "sqlite:///:memory:"

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

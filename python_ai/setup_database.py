"""
Database Setup Script
Run this to create all database tables
"""
from python_ai.models_db import Market, StockDetail, News, PredictionAccuracy, ModelPerformance
from python_ai.database import Base, engine, test_connection
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables"""
    try:
        logger.info("=" * 70)
        logger.info("DATABASE SETUP - STOCK PREDICTION API")
        logger.info("=" * 70)
        
        # Test connection first
        logger.info("\n1. Testing database connection...")
        if not test_connection():
            logger.error("❌ Cannot connect to database!")
            logger.error("   Please check:")
            logger.error("   - MySQL is running")
            logger.error("   - Database 'stockdb' exists")
            logger.error("   - Credentials in .env are correct")
            sys.exit(1)
        
        # Create tables
        logger.info("\n2. Creating tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created successfully!")
        
        # List created tables
        logger.info("\n3. Created tables:")
        for table in Base.metadata.sorted_tables:
            logger.info(f"   ✓ {table.name}")
            
            # Show column count
            col_count = len(table.columns)
            logger.info(f"     └─ {col_count} columns")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ DATABASE SETUP COMPLETE!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Verify .env file has correct API keys")
        logger.info("2. Run: python main.py")
        logger.info("3. Open: http://localhost:8000")
        logger.info("4. Check docs: http://localhost:8000/docs")
        logger.info("\n")
            
    except Exception as e:
        logger.error(f"\n❌ Error creating tables: {str(e)}")
        logger.error("Please check your database configuration")
        sys.exit(1)

def drop_tables():
    """Drop all tables (DANGEROUS - use with caution!)"""
    logger.warning("⚠️  WARNING: This will delete ALL data!")
    response = input("Are you sure? Type 'yes' to continue: ")
    
    if response.lower() == 'yes':
        try:
            Base.metadata.drop_all(bind=engine)
            logger.info("✅ All tables dropped successfully")
        except Exception as e:
            logger.error(f"❌ Error dropping tables: {str(e)}")
    else:
        logger.info("Operation cancelled")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        drop_tables()
    else:
        create_tables()
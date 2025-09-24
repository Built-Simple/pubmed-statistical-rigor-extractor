# test_setup.py - Run this first to verify everything works

import psycopg2
import os
from pathlib import Path

def test_setup():
    """Test all components before running main extraction"""

    print("Testing PubMed Statistical Analysis Setup...")
    print("=" * 50)

    # Test database connection
    print("\n1. Testing database connection...")
    db_config = {
        'host': 'localhost',
        'database': 'pmc_fulltext',   # PubMed Central database
        'user': 'postgres',           # PostgreSQL username
        'password': 'Tapane2001!',    # PostgreSQL password
        'port': 5432
    }

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles WHERE full_text IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"   [OK] Connected! Found {count:,} articles with text")

        # Get sample article
        cursor.execute("SELECT id, LEFT(full_text, 500) FROM articles WHERE full_text IS NOT NULL LIMIT 1")
        sample = cursor.fetchone()
        print(f"   [OK] Sample article ID: {sample[0]}")
        print(f"   [OK] Sample text preview: {sample[1][:100]}...")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   [FAIL] Database connection failed: {e}")
        return False

    # Test directory creation
    print("\n2. Testing directory access...")
    directories = [
        r"D:\PubMed_Statistical_Analysis",
        r"C:\Temp\PubMed_Analysis"
    ]

    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            test_file = os.path.join(directory, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"   [OK] {directory} - OK")
        except Exception as e:
            print(f"   [FAIL] {directory} - Failed: {e}")
            return False

    # Test multiprocessing
    print("\n3. Testing multiprocessing...")
    import multiprocessing as mp
    print(f"   [OK] Available CPU cores: {mp.cpu_count()}")
    print(f"   [OK] Will use 20 processes (leaving 4 threads for system)")

    print("\n" + "=" * 50)
    print("All tests passed! Ready to run extraction.")
    print("\nNext steps:")
    print("1. Update database credentials in fast_statistical_extractor.py")
    print("2. Run: python fast_statistical_extractor.py")

    return True

if __name__ == "__main__":
    test_setup()
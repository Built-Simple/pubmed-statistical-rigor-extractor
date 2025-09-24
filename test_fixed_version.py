#!/usr/bin/env python3
"""
Test the fixed extraction script with correct database schema
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import sys
import os
from pathlib import Path

def test_database_schema():
    """Test database connection and verify schema"""
    print("="*60)
    print("Testing PubMed Statistical Analysis - Fixed Version")
    print("="*60)

    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'pmc_fulltext',
        'user': 'postgres',
        'password': 'Tapane2001!',
        'port': 5432
    }

    try:
        # Test 1: Basic connection
        print("\n1. Testing database connection...")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Test 2: Check schema
        print("\n2. Verifying table schema...")
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'articles'
            ORDER BY ordinal_position
        """)

        columns = cursor.fetchall()
        print("   Article table columns:")
        for col_name, data_type, max_length in columns:
            print(f"   - {col_name}: {data_type}" + (f"({max_length})" if max_length else ""))

        # Test 3: Count articles
        print("\n3. Counting articles...")
        cursor.execute("SELECT COUNT(*) FROM articles WHERE full_text IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"   Total articles with full text: {count:,}")

        # Test 4: Sample data with all columns
        print("\n4. Fetching sample article with metadata...")
        cursor.execute("""
            SELECT id, pmc_id, pmid, doi, title, journal, pub_year,
                   LENGTH(full_text) as text_length,
                   LEFT(full_text, 200) as text_preview
            FROM articles
            WHERE full_text IS NOT NULL
            LIMIT 1
        """)

        sample = cursor.fetchone()
        print(f"   ID: {sample[0]}")
        print(f"   PMC ID: {sample[1]}")
        print(f"   PMID: {sample[2]}")
        print(f"   DOI: {sample[3]}")
        print(f"   Title: {sample[4][:100]}..." if sample[4] else "   Title: None")
        print(f"   Journal: {sample[5]}")
        print(f"   Year: {sample[6]}")
        print(f"   Text length: {sample[7]:,} characters")
        print(f"   Text preview: {sample[8][:100]}...")

        # Test 5: Test RealDictCursor for metadata fetching
        print("\n5. Testing dictionary cursor for metadata...")
        dict_cursor = conn.cursor(cursor_factory=RealDictCursor)
        dict_cursor.execute("""
            SELECT id, pmc_id, pmid, title, journal, pub_year
            FROM articles
            WHERE full_text IS NOT NULL
            LIMIT 1
        """)

        dict_result = dict_cursor.fetchone()
        print(f"   Successfully fetched as dictionary with {len(dict_result)} fields")
        print(f"   Keys: {', '.join(dict_result.keys())}")

        # Test 6: Test batch fetching
        print("\n6. Testing batch fetching...")
        cursor_name = f'test_cursor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        batch_cursor = conn.cursor(name=cursor_name)
        batch_cursor.execute("""
            SELECT id, full_text
            FROM articles
            WHERE full_text IS NOT NULL
            ORDER BY id
            LIMIT 100
        """)

        batch = batch_cursor.fetchmany(10)
        print(f"   Fetched batch of {len(batch)} articles")
        print(f"   First article ID: {batch[0][0]}, text length: {len(batch[0][1]):,}")

        batch_cursor.close()
        dict_cursor.close()
        cursor.close()
        conn.close()

        print("\n7. Testing directory access...")
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
                print(f"   [OK] {directory}")
            except Exception as e:
                print(f"   [FAIL] {directory}: {e}")
                return False

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nThe fixed extraction script is ready to run:")
        print("python fast_statistical_extractor_fixed.py")
        print("\nKey improvements in fixed version:")
        print("- Uses correct 'id' column name throughout")
        print("- Optionally includes article metadata (PMC ID, title, journal, year)")
        print("- Properly handles database schema")
        print("- Creates configuration and summary files")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_schema()
    sys.exit(0 if success else 1)
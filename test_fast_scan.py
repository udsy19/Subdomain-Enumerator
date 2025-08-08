#!/usr/bin/env python3
"""
Test script for fast scanning and parallel domain scanning
"""

from main_tui_merged import fast_scan, multi_domain_scan, quick_compare_domains

def test_fast_scan():
    """Test the fast scanning functionality"""
    print("🧪 Testing fast scan functionality...")
    try:
        # Test with a small wordlist for speed
        results = fast_scan('example.com', wordlist_files=['wordlists/common.txt'])
        print(f"✅ Fast scan test passed: {len(results)} results")
        return True
    except Exception as e:
        print(f"❌ Fast scan test failed: {e}")
        return False

def test_multi_domain():
    """Test multi-domain scanning"""
    print("🧪 Testing multi-domain scan functionality...")
    try:
        domains = ['example.com', 'google.com']
        results = multi_domain_scan(domains, mode=1)
        print(f"✅ Multi-domain test passed: {len(results)} domains scanned")
        return True
    except Exception as e:
        print(f"❌ Multi-domain test failed: {e}")
        return False

def test_compare():
    """Test domain comparison"""
    print("🧪 Testing domain comparison functionality...")
    try:
        results = quick_compare_domains('example.com', 'google.com')
        print(f"✅ Comparison test passed")
        return True
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running fast scan tests...")
    
    # Run tests
    tests = [test_fast_scan, test_multi_domain, test_compare]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
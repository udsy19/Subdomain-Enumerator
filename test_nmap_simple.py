#!/usr/bin/env python3
"""
Simple test to debug Nmap integration issues
"""

import sys
import subprocess
from main_tui_merged import UltraRobustEnumerator

def test_nmap_direct():
    """Test direct Nmap execution"""
    print("🔍 Testing Direct Nmap Execution")
    print("=" * 50)
    
    # Test direct nmap command
    cmd = ['nmap', '-sS', '-T4', '-Pn', '--top-ports', '10', 'google.com']
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_nmap_function():
    """Test the perform_nmap_scan function"""
    print("\n🔍 Testing perform_nmap_scan Function")
    print("=" * 50)
    
    enumerator = UltraRobustEnumerator()
    
    try:
        result = enumerator.perform_nmap_scan("google.com")
        
        print("📊 Function returned:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_nmap_parsing():
    """Test nmap output parsing"""
    print("\n🔍 Testing Nmap Output Parsing")
    print("=" * 50)
    
    # Sample nmap output
    sample_output = """Starting Nmap 7.95 ( https://nmap.org ) at 2025-01-15 10:00 PST
Nmap scan report for google.com (142.250.80.14)
Host is up (0.015s latency).
Not shown: 996 filtered tcp ports (no-responses)
PORT    STATE SERVICE
80/tcp  open  http
443/tcp open  https
8080/tcp closed http-proxy
8443/tcp closed https-alt

Nmap done: 1 IP address (1 host up) scanned in 4.32 seconds"""
    
    enumerator = UltraRobustEnumerator()
    
    try:
        result = enumerator._parse_nmap_output(sample_output)
        
        print("📊 Parsed result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
        return len(result['open_ports']) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple Nmap Debug Test...")
    
    all_passed = True
    
    # Test 1: Direct nmap execution
    if not test_nmap_direct():
        all_passed = False
        
    # Test 2: Function wrapper
    if not test_nmap_function():
        all_passed = False
        
    # Test 3: Output parsing
    if not test_nmap_parsing():
        all_passed = False
        
    print(f"\n{'🎉 All tests passed!' if all_passed else '❌ Some tests failed.'}")
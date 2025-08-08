#!/usr/bin/env python3
"""
Detailed SSL functionality test for Ultra-Robust Subdomain Enumerator
Tests SSL certificate verification across multiple domains and methods
"""

import asyncio
import ssl
import socket
import sys
import time
from typing import Dict, List
from main_tui_merged import UltraRobustEnumerator

# Test domains with known SSL certificates
TEST_DOMAINS = [
    "google.com",
    "github.com", 
    "stackoverflow.com",
    "cloudflare.com",
    "amazon.com",
    "microsoft.com",
    "api.github.com",
    "www.google.com"
]

async def test_ssl_methods_comprehensive():
    """Comprehensive test of all SSL methods"""
    print("🔍 SSL Functionality Comprehensive Test")
    print("=" * 60)
    
    # Initialize the enumerator
    enumerator = UltraRobustEnumerator()
    
    results = {}
    
    for domain in TEST_DOMAINS:
        print(f"\n🎯 Testing SSL for: {domain}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Test the robust SSL info function
            ssl_info = await enumerator._get_ssl_info_robust(domain)
            
            elapsed = time.time() - start_time
            
            if ssl_info:
                print(f"✅ SSL data retrieved in {elapsed:.2f}s")
                print(f"   Issuer: {ssl_info.get('issuer', 'N/A')}")
                print(f"   Subject: {ssl_info.get('subject', 'N/A')}")
                print(f"   Expiry: {ssl_info.get('expiry', 'N/A')}")
                print(f"   Serial: {ssl_info.get('serial', 'N/A')}")
                san_domains = ssl_info.get('san_domains', [])
                if san_domains:
                    print(f"   SAN Domains: {', '.join(san_domains[:3])}{'...' if len(san_domains) > 3 else ''}")
                
                results[domain] = {
                    'status': 'success',
                    'issuer': ssl_info.get('issuer'),
                    'subject': ssl_info.get('subject'),
                    'time': elapsed
                }
            else:
                print(f"❌ SSL data retrieval failed in {elapsed:.2f}s")
                results[domain] = {
                    'status': 'failed',
                    'time': elapsed
                }
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Exception during SSL test: {str(e)}")
            results[domain] = {
                'status': 'error',
                'error': str(e),
                'time': elapsed
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SSL Test Summary")
    print("=" * 60)
    
    successful = [d for d, r in results.items() if r['status'] == 'success']
    failed = [d for d, r in results.items() if r['status'] in ['failed', 'error']]
    
    print(f"✅ Successful: {len(successful)}/{len(TEST_DOMAINS)} domains")
    print(f"❌ Failed: {len(failed)}/{len(TEST_DOMAINS)} domains")
    
    if successful:
        avg_time = sum(results[d]['time'] for d in successful) / len(successful)
        print(f"⏱️  Average retrieval time: {avg_time:.2f}s")
    
    if failed:
        print(f"\n❌ Failed domains: {', '.join(failed)}")
        
    return results

async def test_direct_ssl_connection():
    """Test direct SSL connection method"""
    print("\n🔧 Direct SSL Connection Test")
    print("-" * 40)
    
    for domain in TEST_DOMAINS[:3]:  # Test first 3 domains
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert_dict = ssock.getpeercert()
                    cert_binary = ssock.getpeercert(binary_form=True)
                    
                    print(f"🔍 {domain}:")
                    print(f"   cert_dict type: {type(cert_dict)}")
                    print(f"   cert_dict length: {len(cert_dict) if cert_dict else 0}")
                    print(f"   cert_binary length: {len(cert_binary) if cert_binary else 0}")
                    
                    if cert_dict:
                        print(f"   cert_dict keys: {list(cert_dict.keys())[:5]}")
                    
        except Exception as e:
            print(f"❌ {domain}: {str(e)}")

def test_cryptography_import():
    """Test cryptography library availability"""
    print("\n📚 Cryptography Library Test")
    print("-" * 40)
    
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        print("✅ Cryptography library is available")
        print(f"   x509 module: {x509}")
        print(f"   default_backend: {default_backend}")
        return True
    except ImportError as e:
        print(f"❌ Cryptography library not available: {e}")
        return False

async def main():
    """Main test runner"""
    print("🚀 Ultra-Robust Subdomain Enumerator SSL Test Suite")
    print("=" * 70)
    
    # Test 1: Cryptography import
    crypto_available = test_cryptography_import()
    
    # Test 2: Direct SSL connection
    await test_direct_ssl_connection()
    
    # Test 3: Comprehensive SSL test
    results = await test_ssl_methods_comprehensive()
    
    # Final assessment
    print("\n" + "=" * 70)
    print("🏁 Final Assessment")
    print("=" * 70)
    
    successful_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"📈 Overall Success Rate: {success_rate:.1f}% ({successful_count}/{total_count})")
    
    if success_rate >= 80:
        print("✅ SSL functionality is working well!")
    elif success_rate >= 50:
        print("⚠️  SSL functionality has some issues but partially working")
    else:
        print("❌ SSL functionality needs significant fixes")
    
    if not crypto_available:
        print("💡 Consider installing cryptography library: pip install cryptography")

if __name__ == "__main__":
    asyncio.run(main())
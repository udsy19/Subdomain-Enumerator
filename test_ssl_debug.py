#!/usr/bin/env python3
"""
Debug script to test SSL certificate analysis
"""

import asyncio
import ssl
import socket
import subprocess
from typing import Dict, List

# Check if cryptography is available
try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
    print("✅ Cryptography library is available")
except ImportError:
    CRYPTO_AVAILABLE = False
    print("❌ Cryptography library is NOT available")

async def test_ssl_analysis(hostname: str):
    """Test SSL analysis with debug output"""
    print(f"\n🔍 Testing SSL analysis for: {hostname}")
    
    # Method 1: Direct SSL connection
    print("\n--- Method 1: Direct SSL connection ---")
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        print(f"Attempting to connect to {hostname}:443...")
        with socket.create_connection((hostname, 443), timeout=10) as sock:
            print("✅ Socket connection established")
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                print("✅ SSL handshake successful")
                # Try to get certificate with binary form first
                cert_der = ssock.getpeercert(binary_form=True)
                print(f"Binary certificate length: {len(cert_der) if cert_der else 'None'}")
                
                # Now get the parsed certificate
                cert_dict = ssock.getpeercert()
                print(f"Certificate dict type: {type(cert_dict)}")
                print(f"Certificate dict keys: {list(cert_dict.keys()) if cert_dict else 'None'}")
                
                if cert_dict:
                    print("✅ Certificate obtained")
                    issuer_dict = dict(cert_dict.get('issuer', []))
                    subject_dict = dict(cert_dict.get('subject', []))
                    
                    print(f"Issuer dict: {issuer_dict}")
                    print(f"Subject dict: {subject_dict}")
                    
                    issuer = issuer_dict.get('organizationName', 'Unknown Issuer')
                    subject = subject_dict.get('commonName', hostname)
                    
                    print(f"Extracted issuer: {issuer}")
                    print(f"Extracted subject: {subject}")
                    
                    if 'subjectAltName' in cert_dict:
                        san_list = cert_dict['subjectAltName']
                        print(f"SAN found: {san_list}")
                    else:
                        print("No SAN found")
                        
                    return {
                        'method': 'direct_ssl',
                        'issuer': issuer,
                        'subject': subject,
                        'success': True
                    }
                else:
                    print("❌ No certificate obtained from getpeercert()")
                    # Try to use the binary certificate with cryptography
                    if cert_der and CRYPTO_AVAILABLE:
                        print("Trying to parse binary certificate...")
                        try:
                            cert = x509.load_der_x509_certificate(cert_der, default_backend())
                            
                            # Extract issuer
                            issuer = "Unknown"
                            for attribute in cert.issuer:
                                if attribute.oid == x509.NameOID.ORGANIZATION_NAME:
                                    issuer = attribute.value
                                    break
                            
                            # Extract subject
                            subject = hostname
                            for attribute in cert.subject:
                                if attribute.oid == x509.NameOID.COMMON_NAME:
                                    subject = attribute.value
                                    break
                            
                            print(f"✅ Binary cert parsed - Issuer: {issuer}, Subject: {subject}")
                            return {
                                'method': 'direct_ssl_binary',
                                'issuer': issuer,
                                'subject': subject,
                                'success': True
                            }
                        except Exception as parse_e:
                            print(f"❌ Failed to parse binary certificate: {parse_e}")
                    
                    return {'method': 'direct_ssl', 'success': False, 'error': 'No certificate'}
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Using cryptography library
    if CRYPTO_AVAILABLE:
        print("\n--- Method 2: Cryptography library ---")
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((hostname, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
                    cert = x509.load_der_x509_certificate(cert_der, default_backend())
                    
                    # Extract issuer
                    issuer = "Unknown"
                    for attribute in cert.issuer:
                        if attribute.oid == x509.NameOID.ORGANIZATION_NAME:
                            issuer = attribute.value
                            break
                    
                    # Extract subject
                    subject = hostname
                    for attribute in cert.subject:
                        if attribute.oid == x509.NameOID.COMMON_NAME:
                            subject = attribute.value
                            break
                    
                    print(f"✅ Method 2 successful - Issuer: {issuer}, Subject: {subject}")
                    return {
                        'method': 'cryptography',
                        'issuer': issuer,
                        'subject': subject,
                        'success': True
                    }
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
    
    # Method 3: OpenSSL command
    print("\n--- Method 3: OpenSSL command ---")
    try:
        cmd = ['openssl', 's_client', '-connect', f'{hostname}:443', '-servername', hostname, '-verify_return_error']
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, input='', capture_output=True, text=True, timeout=15)
        
        print(f"Command return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)} chars")
        print(f"STDERR length: {len(result.stderr)} chars")
        
        if result.returncode == 0 or 'CERTIFICATE' in result.stdout:
            print("✅ OpenSSL command successful")
            
            # Parse output
            ssl_info = {'issuer': 'Unknown', 'subject': 'Unknown'}
            lines = result.stdout.split('\n')
            for line in lines:
                if 'issuer=' in line:
                    print(f"Found issuer line: {line}")
                    parts = line.split('issuer=')[1].split(',')
                    for part in parts:
                        if 'O=' in part:
                            ssl_info['issuer'] = part.split('O=')[1].strip()
                            break
                elif 'subject=' in line:
                    print(f"Found subject line: {line}")
                    parts = line.split('subject=')[1].split(',')
                    for part in parts:
                        if 'CN=' in part:
                            ssl_info['subject'] = part.split('CN=')[1].strip()
                            break
            
            print(f"Parsed SSL info: {ssl_info}")
            return {
                'method': 'openssl',
                'issuer': ssl_info['issuer'],
                'subject': ssl_info['subject'],
                'success': True
            }
        else:
            print(f"❌ OpenSSL command failed")
            print(f"STDERR: {result.stderr[:200]}")
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    print("❌ All methods failed")
    return {'success': False, 'error': 'All methods failed'}

async def main():
    """Test SSL analysis on multiple domains"""
    test_domains = [
        'google.com',
        'github.com', 
        'stackoverflow.com',
        'cloudflare.com'
    ]
    
    for domain in test_domains:
        result = await test_ssl_analysis(domain)
        print(f"\n{'='*50}")
        print(f"Final result for {domain}: {result}")
        print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
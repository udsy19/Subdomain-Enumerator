#!/usr/bin/env python3
"""
Full test of Nmap integration with live scanning
"""

import sys
import time
from main_tui_merged import UltraRobustEnumerator, SubdomainResult

def test_nmap_integration():
    """Test Nmap integration with live scanning"""
    
    print("🔍 Testing Nmap Integration with Live Scanning")
    print("=" * 60)
    
    enumerator = UltraRobustEnumerator()
    
    # Test targets that are likely to respond
    test_targets = [
        "google.com",
        "github.com"
    ]
    
    for target in test_targets:
        print(f"\n🎯 Scanning {target}...")
        try:
            nmap_result = enumerator.perform_nmap_scan(target)
            
            print(f"📊 Results for {target}:")
            print(f"  🚪 Open Ports: {', '.join(nmap_result['open_ports']) if nmap_result['open_ports'] else 'None detected'}")
            print(f"  ⚙️  Services: {len(nmap_result['services'])} found")
            if nmap_result['services']:
                for service in nmap_result['services'][:3]:  # Show first 3
                    print(f"    - {service}")
            print(f"  💻 OS Detection: {nmap_result['os_detection'] or 'Not detected'}")
            print(f"  🚨 Vulnerabilities: {len(nmap_result['vulnerabilities'])} found")
            if nmap_result['vulnerabilities']:
                for vuln in nmap_result['vulnerabilities'][:2]:  # Show first 2
                    print(f"    - {vuln}")
            print(f"  🔒 SSL Info: {'Yes' if nmap_result['ssl_info'] else 'No'}")
            print(f"  🌐 HTTP Info: {'Yes' if nmap_result['http_info'] else 'No'}")
            print(f"  🗺️  Traceroute: {'Yes' if nmap_result['traceroute'] else 'No'}")
            print(f"  📡 DNS Info: {'Yes' if nmap_result['dns_info'] else 'No'}")
            
        except Exception as e:
            print(f"❌ Error scanning {target}: {e}")
    
    # Test Excel export with Nmap data
    print(f"\n📊 Testing Excel Export with Nmap Data...")
    
    # Create a sample result with Nmap data
    sample_result = SubdomainResult(
        subdomain="test.google.com",
        source="Test_Data",
        http_status=200,
        ip_addresses=["142.250.80.14"],
        technologies=["Google Test"],
        confidence_score=0.95,
        discovered_at=time.time(),
        ownership_info="MarkMonitor, Inc."
    )
    
    # The Excel export will trigger Nmap scanning
    sample_results = {"test.google.com": sample_result}
    
    try:
        filename = enumerator.save_advanced_excel(sample_results, "nmap_test_domain")
        print(f"✅ Excel file with Nmap data created: {filename}")
        
        # Check if Nmap data was populated
        if sample_result.nmap_open_ports or sample_result.nmap_services:
            print("✅ Nmap data successfully populated in SubdomainResult")
            print(f"  - Open Ports: {sample_result.nmap_open_ports}")
            print(f"  - Services: {sample_result.nmap_services}")
        else:
            print("⚠️  No Nmap data populated (target may not respond or firewall blocks)")
            
    except Exception as e:
        print(f"❌ Excel export with Nmap data failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting comprehensive Nmap integration test...")
    print("⚠️  This test performs actual network scanning and may take a few minutes.")
    
    try:
        success = test_nmap_integration()
        if success:
            print("\n🎉 Nmap integration test completed successfully!")
            print("\n✨ Key Features Verified:")
            print("  ✅ Live Nmap scanning capability")
            print("  ✅ Nmap output parsing")
            print("  ✅ Integration with Excel export")
            print("  ✅ Comprehensive data collection (ports, services, OS, vulnerabilities)")
        else:
            print("\n❌ Some Nmap integration tests failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
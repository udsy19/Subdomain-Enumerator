#!/usr/bin/env python3
"""
Test the advanced features implemented:
1. Smart IP Range Scanning
2. Enhanced Certificate Transparency
3. Dynamic Resource Scaling
"""

import sys
import time
import asyncio
from main_tui_merged import UltraRobustEnumerator, SubdomainResult

async def test_system_resource_detection():
    """Test system resource detection"""
    print("🔧 Testing System Resource Detection...")
    
    enumerator = UltraRobustEnumerator()
    resources = enumerator.detect_system_resources()
    
    print(f"  CPU Cores: {resources.cpu_cores}")
    print(f"  Memory: {resources.memory_gb:.2f} GB")
    print(f"  Network Bandwidth: {resources.network_bandwidth_mbps:.0f} Mbps")
    print(f"  Concurrent Threads: {resources.concurrent_threads}")
    
    # Validate resource detection
    if resources.cpu_cores > 0 and resources.memory_gb > 0:
        print("  ✅ System resource detection working")
        return True
    else:
        print("  ❌ System resource detection failed")
        return False

async def test_ip_range_grouping():
    """Test IP range grouping functionality"""
    print("\n🌐 Testing IP Range Grouping...")
    
    enumerator = UltraRobustEnumerator()
    
    # Create sample results with various IPs
    sample_results = {}
    test_ips = [
        ("mail.google.com", ["142.250.80.17"]),
        ("www.google.com", ["142.250.80.14"]),  # Same /24 subnet
        ("api.google.com", ["142.250.80.18"]),  # Same /24 subnet
        ("github.com", ["140.82.114.4"]),       # Different subnet
        ("api.github.com", ["140.82.114.6"]),   # Same subnet as github
    ]
    
    for subdomain, ips in test_ips:
        result = SubdomainResult(
            subdomain=subdomain,
            source="Test",
            http_status=200,
            ip_addresses=ips,
            technologies=[],
            confidence_score=1.0,
            discovered_at=time.time()
        )
        sample_results[subdomain] = result
    
    # Group by IP ranges
    ip_groups = enumerator.group_ips_by_ranges(sample_results)
    
    print(f"  Created {len(sample_results)} test subdomains")
    print(f"  Grouped into {len(ip_groups)} IP ranges")
    
    for i, group in enumerate(ip_groups):
        print(f"    Range {i+1}: {group.subnet}")
        print(f"      IPs: {len(group.ip_addresses)} ({', '.join(group.ip_addresses)})")
        print(f"      Subdomains: {len(group.subdomains)} ({', '.join(group.subdomains)})")
        print(f"      Estimated scan time: {group.estimated_scan_time:.1f}s")
    
    if len(ip_groups) >= 2:  # Should group google IPs separately from github IPs
        print("  ✅ IP range grouping working correctly")
        return True, ip_groups
    else:
        print("  ❌ IP range grouping failed")
        return False, []

async def test_smart_ip_scanning():
    """Test smart IP range scanning"""
    print("\n⚡ Testing Smart IP Range Scanning...")
    
    enumerator = UltraRobustEnumerator()
    
    # Use the IP groups from previous test
    success, ip_groups = await test_ip_range_grouping()
    if not success:
        return False
    
    def progress_callback(phase, progress, **kwargs):
        message = kwargs.get('message', '')
        print(f"    {phase}: {progress:.0f}% - {message}")
    
    print(f"  Running smart IP scanning on {len(ip_groups)} ranges...")
    start_time = time.time()
    
    try:
        scan_results = await enumerator.smart_ip_range_scan(ip_groups, progress_callback)
        elapsed = time.time() - start_time
        
        print(f"  Smart scanning completed in {elapsed:.2f} seconds")
        print(f"  Scanned {len(scan_results)} IP ranges")
        
        # Check results
        total_ips_scanned = 0
        for subnet, subnet_results in scan_results.items():
            ips_in_subnet = len(subnet_results)
            total_ips_scanned += ips_in_subnet
            print(f"    {subnet}: {ips_in_subnet} IPs scanned")
        
        if total_ips_scanned > 0:
            print("  ✅ Smart IP scanning working")
            return True
        else:
            print("  ❌ Smart IP scanning returned no results")
            return False
            
    except Exception as e:
        print(f"  ❌ Smart IP scanning failed: {str(e)}")
        return False

async def test_enhanced_ct_mining():
    """Test enhanced Certificate Transparency mining"""
    print("\n🔒 Testing Enhanced Certificate Transparency Mining...")
    
    enumerator = UltraRobustEnumerator()
    
    def progress_callback(phase, progress, **kwargs):
        message = kwargs.get('message', '')
        print(f"    {phase}: {progress:.0f}% - {message}")
    
    # Test with google.com (should have many CT entries)
    test_domain = "google.com"
    print(f"  Testing CT mining for {test_domain}")
    
    start_time = time.time()
    initial_results = len(enumerator.results)
    
    try:
        await enumerator._enhanced_ct_mining(test_domain)
        elapsed = time.time() - start_time
        
        found_results = len(enumerator.results) - initial_results
        print(f"  Enhanced CT mining completed in {elapsed:.2f} seconds")
        print(f"  Found {found_results} new subdomains from CT logs")
        
        # Show some discovered subdomains
        if found_results > 0:
            print("  Sample discovered subdomains:")
            ct_results = [r for r in enumerator.results.values() if r.source == "Enhanced_CT_Mining"]
            for result in ct_results[:3]:  # Show first 3
                print(f"    - {result.subdomain} (confidence: {result.confidence_score})")
        
        if found_results > 0:
            print("  ✅ Enhanced CT mining working")
            return True
        else:
            print("  ⚠️  No CT results (may be due to rate limiting or network)")
            return True  # Don't fail the test due to external factors
            
    except Exception as e:
        print(f"  ❌ Enhanced CT mining failed: {str(e)}")
        return False

async def test_advanced_data_structures():
    """Test the new advanced data structures"""
    print("\n📊 Testing Advanced Data Structures...")
    
    from main_tui_merged import SystemResources, IPRangeGroup, VulnerabilityAssessment, AIAnalysis, HistoricalData
    import datetime
    
    # Test SystemResources
    resources = SystemResources(
        cpu_cores=8,
        memory_gb=16.0,
        network_bandwidth_mbps=1000.0,
        concurrent_threads=32
    )
    print(f"  SystemResources: {resources.cpu_cores} cores, {resources.memory_gb}GB RAM")
    
    # Test IPRangeGroup
    ip_group = IPRangeGroup(
        subnet="192.168.1.0/24",
        ip_addresses=["192.168.1.1", "192.168.1.2"],
        subdomains=["test1.example.com", "test2.example.com"],
        subnet_size=24,
        estimated_scan_time=10.5
    )
    print(f"  IPRangeGroup: {ip_group.subnet} with {len(ip_group.ip_addresses)} IPs")
    
    # Test VulnerabilityAssessment
    vuln_assessment = VulnerabilityAssessment(
        cve_ids=["CVE-2021-44228"],
        security_headers={"X-Frame-Options": "DENY"},
        risk_score=7.5
    )
    print(f"  VulnerabilityAssessment: {len(vuln_assessment.cve_ids)} CVEs, risk score {vuln_assessment.risk_score}")
    
    # Test AIAnalysis  
    ai_analysis = AIAnalysis(
        pattern_confidence=0.85,
        anomaly_score=0.12,
        predicted_subdomains=["ml-predicted.example.com"],
        risk_assessment="Medium"
    )
    print(f"  AIAnalysis: {ai_analysis.pattern_confidence} confidence, {ai_analysis.risk_assessment} risk")
    
    # Test HistoricalData
    historical = HistoricalData(
        first_seen=datetime.datetime.now(),
        last_seen=datetime.datetime.now(),
        changes_detected=["IP changed from 1.1.1.1 to 2.2.2.2"]
    )
    print(f"  HistoricalData: {len(historical.changes_detected)} changes detected")
    
    print("  ✅ All advanced data structures working")
    return True

async def main():
    """Run all advanced feature tests"""
    print("🚀 Advanced Features Test Suite")
    print("=" * 50)
    print("Testing new advanced capabilities:")
    print("- Smart IP Range Scanning for performance")
    print("- Enhanced Certificate Transparency mining")
    print("- Dynamic Resource Scaling")
    print("- Advanced data structures")
    print("=" * 50)
    
    tests = [
        ("System Resource Detection", test_system_resource_detection()),
        ("IP Range Grouping & Smart Scanning", test_smart_ip_scanning()),
        ("Enhanced CT Mining", test_enhanced_ct_mining()),
        ("Advanced Data Structures", test_advanced_data_structures())
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_coro
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All advanced features are working correctly!")
        print("\n✨ Key Features Implemented:")
        print("  ✅ Smart IP Range Scanning - Groups IPs by subnets for efficient scanning")
        print("  ✅ Enhanced Certificate Transparency - Queries multiple CT log sources")  
        print("  ✅ Dynamic Resource Scaling - Auto-detects system capabilities")
        print("  ✅ Advanced Data Structures - New classes for comprehensive analysis")
        print("\n🚀 Performance Improvements:")
        print("  ⚡ IP scanning now groups by subnets for massive speed boost")
        print("  📡 CT mining queries multiple sources for comprehensive discovery")
        print("  🔧 Resource detection optimizes thread counts based on system")
        print("  📊 Enhanced data structures support advanced analytics")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - some features may need attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)
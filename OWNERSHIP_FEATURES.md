# Domain Ownership Detection & Nmap Integration Features

## Overview
Enhanced the Ultra-Robust Subdomain Enumerator with domain ownership detection capabilities using WHOIS lookups, comprehensive Nmap scanning, and advanced Excel reporting with detailed network intelligence.

## New Features Added

### 1. Domain Ownership Detection
- **Function**: `get_domain_ownership(domain_or_subdomain: str) -> str`
- **Location**: `main_tui_merged.py` line ~2289
- **Capabilities**:
  - Extracts root domain from subdomains automatically
  - Uses python-whois library as primary method
  - Falls back to system `whois` command if library unavailable
  - Returns registrar or organization information
  - Handles errors gracefully with "Unknown" fallback

### 2. Enhanced Data Structure
- **Modified**: `SubdomainResult` dataclass
- **New Field**: `ownership_info: Optional[str] = None`
- **Purpose**: Stores ownership/registrar information for each subdomain

### 3. Enhanced Excel Export - Main Sheet
- **New Column**: "Domain_Owner" 
- **Position**: Between "Discovery_Source" and "HTTP_Status"
- **Content**: Domain registrar/organization information
- **Auto-population**: Automatically fetches ownership if not already populated

### 4. Nmap Integration for Comprehensive Network Intelligence
- **Function**: `perform_nmap_scan(target: str) -> Dict[str, any]`
- **Location**: `main_tui_merged.py` line ~2343
- **Capabilities**:
  - **Open Ports Detection**: TCP/UDP ports that are open
  - **Service Version Detection**: Exact versions of services (Apache, SSH, etc.)
  - **Operating System Detection**: OS fingerprinting and version identification
  - **Vulnerability Scanning**: CVE detection using NSE scripts
  - **SSL/TLS Analysis**: Certificate details, ciphers, protocol support
  - **HTTP Information**: Server headers, titles, directory enumeration
  - **Network Traceroute**: Path analysis to target
  - **DNS Intelligence**: Zone transfer attempts, DNS analysis
- **Command Used**: `nmap -A -sV -sS -O --script vuln,ssl-cert,ssl-enum-ciphers,http-enum,http-title,http-headers,dns-zone-transfer --traceroute -T4 -Pn`
- **Timeout Protection**: 5-minute timeout with graceful error handling

### 5. Enhanced SubdomainResult Data Structure
- **New Nmap Fields Added**:
  - `nmap_open_ports: List[str]` - List of open ports
  - `nmap_services: List[str]` - Service details with versions
  - `nmap_os_detection: Optional[str]` - Operating system information
  - `nmap_vulnerabilities: List[str]` - Detected vulnerabilities
  - `nmap_ssl_info: Optional[str]` - SSL/TLS certificate and cipher information
  - `nmap_http_info: Optional[str]` - HTTP server and header information
  - `nmap_traceroute: Optional[str]` - Network path information
  - `nmap_dns_info: Optional[str]` - DNS configuration and zone transfer results

### 6. New Detailed Attributes Sheet
- **Sheet Name**: "Detailed Attributes"
- **Format**: One row per domain/subdomain, one column per attribute
- **SSL Columns Removed**: Old SSL columns replaced with comprehensive Nmap data
- **Columns**: 26 detailed attributes including:
  - Domain/Subdomain
  - Domain_Owner
  - Discovery_Source
  - HTTP_Status & Status_Explanation
  - IP_Addresses
  - Response_Time_ms
  - Technologies
  - Server
  - Confidence_Score
  - CNAME information (Chain, Target, Service Provider/Type)
  - Takeover_Risk
  - Discovered_At timestamp
  - WHOIS_Registrar
  - Root_Domain
  - Subdomain_Level
  - **Nmap_Open_Ports** - Detected open ports
  - **Nmap_Services** - Service details and versions
  - **Nmap_OS_Detection** - Operating system fingerprinting
  - **Nmap_Vulnerabilities** - CVE and security issues found
  - **Nmap_SSL_Info** - SSL certificates and cipher analysis
  - **Nmap_HTTP_Info** - Web server and HTTP header details
  - **Nmap_Traceroute** - Network path analysis
  - **Nmap_DNS_Info** - DNS configuration and zone transfer results
  - Additional_Info

## Usage

### Running the Tool
The ownership detection is automatically integrated into the existing subdomain enumeration process:

```bash
python3 main_tui_merged.py
```

### Manual Testing
Use the test script to verify functionality:

```bash
python3 test_ownership.py
```

### Dependencies
- `python-whois` (optional, will fall back to system command)
- `subprocess` (for system whois command fallback and Nmap execution)
- `openpyxl` (for Excel export)
- `datetime` (for timestamp formatting)
- **`nmap`** (system binary for network scanning)
  - Install on macOS: `brew install nmap`
  - Install on Ubuntu/Debian: `sudo apt install nmap`
  - Install on CentOS/RHEL: `sudo yum install nmap`

## Excel Output Structure

### Sheet 1: "Subdomain Discovery" (Enhanced)
- Original columns plus new "Domain_Owner" column
- Color-coded cells for various risk levels and statuses
- Auto-adjusted column widths

### Sheet 2: "Detailed Attributes" (New)
- Comprehensive per-domain analysis
- 24 attributes per subdomain
- Structured for data analysis and reporting
- Blue header styling for differentiation

## Error Handling
- Graceful handling of WHOIS lookup timeouts
- Fallback mechanisms for unavailable libraries
- "Unknown" returned for failed lookups
- No interruption to main enumeration process

## Performance Considerations
- WHOIS lookups are cached in the `ownership_info` field
- Lookups performed during Excel export to avoid slowing enumeration
- Timeout protection (10 seconds) for WHOIS queries
- Root domain extraction to minimize duplicate lookups

## Security Features
- Defensive design: no execution of untrusted input
- Timeout protection against hanging WHOIS queries
- Error handling prevents tool crashes
- System command execution limited to standard `whois` tool

## Future Enhancements
- Integration with nmap for port scanning data
- Additional WHOIS fields (creation date, expiration, etc.)
- Caching mechanism for repeated domains
- Bulk WHOIS processing optimization
- Integration with threat intelligence feeds

## Testing
The implementation includes a comprehensive test suite (`test_ownership.py`) that verifies:
- WHOIS lookup functionality
- Excel export with ownership data
- Data structure integrity
- Error handling capabilities
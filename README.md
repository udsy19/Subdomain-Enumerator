# Ultra-Robust Subdomain Enumerator v3.0

A powerful, AI-enhanced subdomain enumeration tool designed for comprehensive cybersecurity reconnaissance and defensive security analysis. Now with **11 advanced discovery sources** and **real-time interactive controls**.

## 🚀 New Features (v3.0)

### 🎮 **Interactive Controls**
- **⏸️ SPACE** - Pause/Resume scanning in real-time
- **🛑 Q** - Quit with automatic result saving
- **⌨️ Keyboard Control** - Full keyboard interaction during scanning

### 📊 **Enhanced Progress Tracking**
- **tqdm Integration** - Professional progress bars with time tracking
- **Elapsed/Remaining Time** - Know exactly how long scans take
- **Processing Rate** - Real-time subdomains per second metrics
- **ETA Calculations** - Accurate completion time estimates

### 🔍 **11 Discovery Sources**
1. **Certificate Transparency Mining** - Multiple CT log sources
2. **DNS Intelligence** - 8+ resolver failover system
3. **AI/ML Generation** - 6 different ML prediction techniques
4. **Recursive Discovery** - Automated nested subdomain detection
5. **DNS Zone Transfer** - Advanced DNS enumeration
6. **DNS TXT Mining** - Hidden subdomain discovery
7. **Reverse DNS Lookup** - Infrastructure mapping
8. **DNS ANY Records** - Service discovery
9. **Robots.txt Analysis** - Web crawler hint extraction
10. **Sitemap Analysis** - XML sitemap subdomain extraction
11. **Security.txt Analysis** - Security policy subdomain discovery

### 📈 **Massive Performance Improvements**
- **Lightning Mode**: 1000 DNS workers, 400 HTTP workers (2x faster)
- **Aggressive Mode**: 800 DNS workers, 200 HTTP workers
- **Memory Optimization** - Handles 4+ million subdomain wordlists efficiently
- **Intelligent Batching** - Optimized for massive wordlist processing

### 📋 **Excel Output Enhancements**
- **Readable Source Names** - "Certificate Transparency" instead of "CT_Mining"
- **Professional Formatting** - Color-coded results with confidence indicators
- **Guaranteed Saving** - Results always saved, even when quitting early
- **Enhanced SSL Analysis** - Certificate SAN field extraction

## 🎯 Core Features

- **Multi-Resolver DNS Intelligence** - Uses 8+ public DNS resolvers with intelligent failover
- **Advanced Certificate Transparency Mining** - Queries multiple CT log sources  
- **Machine Learning Predictions** - AI-powered subdomain pattern analysis and prediction
- **Network Infrastructure Analysis** - Reverse DNS, subnet scanning, and ASN analysis
- **Recursive Discovery** - Finds nested subdomains automatically
- **Technology Detection** - Identifies web technologies and server information
- **Real-time Statistics** - Performance metrics and resolver statistics
- **Professional Excel Reports** - Comprehensive Excel output with color coding and statistics

## 📦 Installation

### Requirements

- **Python 3.7+** 
- **Git** (for cloning)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/udsy19/Subdomain-Enumerator.git
cd Subdomain-Enumerator

# Install Python dependencies (will auto-install if missing)
python3 main_tui_merged.py
```

### Manual Installation

```bash
# Install required dependencies
pip install rich tqdm aiohttp aiodns openpyxl cryptography

# Run the tool
python3 main_tui_merged.py
```

## 🎯 Usage

### 🌟 Beautiful TUI Interface

Experience the enhanced terminal UI with real-time interactive controls:

```bash
# Run the enhanced TUI interface
python3 main_tui_merged.py
```

**Interactive Features:**
- 🎨 **Modern Interface** - Gorgeous styling with live animations
- ⏸️ **Pause/Resume** - SPACEBAR to pause/resume scanning anytime
- 🛑 **Smart Quit** - Q key saves results before exiting
- 📊 **tqdm Progress** - Professional progress bars with time tracking
- 📈 **Real-time Stats** - Live discovery rate, ETA, and processing metrics
- 🎯 **Interactive Wordlist Selection** - Choose from 9 different wordlists
- 📱 **Responsive Design** - Adapts to any terminal size

### Configuration Options

The TUI will guide you through these enhanced options:

1. **🎯 Target Domain** - Enter the domain to enumerate (e.g., `example.com`)
2. **🚀 Enumeration Mode**:
   - **Standard** - 400 DNS workers, balanced coverage (recommended)
   - **Aggressive** - 800 DNS workers, maximum coverage
   - **Stealth** - 100 DNS workers, minimal footprint
   - **Lightning** - 1000 DNS workers, maximum speed
3. **📚 Wordlist Selection**:
   - Choose from 9 different wordlists
   - Support for massive 4+ million subdomain wordlists
   - Intelligent memory management for large files

### Real-time Control Commands

**During scanning:**
- **SPACEBAR** - Pause/Resume scanning
- **Q** - Quit and save results
- **Ctrl+C** - Emergency stop with result saving

### Live Progress Example

```
🚀 Ultra-Robust Subdomain Enumeration Starting...
════════════════════════════════════════════════════════════════════════════════

🎧 Controls: SPACE = Pause/Resume, Q = Quit

Overall Progress: 45%|████████████▒▒▒▒▒▒▒▒| 45/100 [00:23<00:28, 1.96%/s]

🔍 www.example.com | Certificate Transparency | 93.184.216.34
🔍 api.example.com | Certificate Transparency | 52.172.141.40
🚨 INTERESTING: api.example.com - Potential security relevance!

💡 Certificate Transparency: CT Mining complete: 2 subdomains found
💡 DNS Brute Force: Batch 5/1795 complete: 0 found, 2 total
DNS Brute Force: Testing batch 6/1795 - 12,000/3,589,968 subdomains tested

⏸️  PAUSED - Press SPACE to continue, Q to quit
▶️  RESUMED

⏱️  Elapsed: 00:23 | ETA: 00:28 | Rate: 145 subdomains/sec
```

## 📁 Enhanced Output

The tool generates comprehensive Excel reports with enhanced features:

- **Filename**: `domain_ultra_robust_YYYYMMDD_HHMMSS.xlsx`
- **Enhanced Columns**:
  - **Discovery_Source** - Human-readable source names
  - **SSL Certificate Analysis** - Full SSL chain verification
  - **Response Time Tracking** - Performance metrics
  - **Technology Detection** - Server and framework identification

### Excel Report Enhancements

- **Readable Sources** - "Certificate Transparency" instead of "CT_Mining"
- **Color-coded HTTP status** - Green=200, Blue=3xx, Red=4xx+
- **Confidence scoring** - High/Medium/Low confidence indicators
- **SSL Certificate SAN** - Subject Alternative Names extraction
- **Guaranteed Saving** - Results saved even with early quit (Q or Ctrl+C)

## 🔧 Enhanced Performance Configuration

The tool now offers significantly improved performance:

| Mode | DNS Workers | HTTP Workers | Timeout | Use Case |
|------|-------------|--------------|---------|----------|
| Standard | 400 | 100 | 5s | Balanced performance |
| Aggressive | 800 | 200 | 8s | Maximum results |
| Stealth | 100 | 50 | 15s | Minimal detection |
| Lightning | 1000 | 400 | 3s | Maximum speed |

## 🔍 Discovery Source Details

### Core Sources (Always Active)
1. **Certificate Transparency** - Scans CT logs for domain certificates
2. **DNS Intelligence** - Multi-resolver DNS queries with failover
3. **AI/ML Generation** - 6 ML techniques including n-gram analysis
4. **Recursive Discovery** - Finds subdomains from discovered subdomains

### Advanced Sources (v3.0)
5. **DNS Zone Transfer** - Attempts zone transfers and DNS enumeration
6. **DNS TXT Mining** - Extracts subdomains from TXT records
7. **Reverse DNS** - Maps IP ranges to find additional subdomains
8. **DNS ANY Records** - Service discovery through ANY queries
9. **Robots.txt Analysis** - Finds admin/dev subdomains from robots files
10. **Sitemap Analysis** - Extracts subdomains from XML sitemaps
11. **Security.txt** - Discovers security-related subdomains

## 🛡️ Security & Ethics

This tool is designed for **defensive security purposes only**:

- ✅ **Authorized penetration testing**
- ✅ **Security assessments of your own domains**
- ✅ **Bug bounty programs with proper authorization**
- ✅ **Academic research and education**

**⚠️ Important**: Only use this tool on domains you own or have explicit written permission to test.

## 🐛 Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
# The tool will offer to auto-install missing packages
# Or install manually:
pip install rich tqdm aiohttp aiodns openpyxl cryptography
```

**Memory Issues with Large Wordlists**
- The tool now uses memory-mapped I/O for files >1MB
- Automatic garbage collection during processing
- Choose smaller wordlists if you encounter issues

**Performance Issues**
- Start with Standard mode (400 workers)
- Use Lightning mode (1000 workers) for maximum speed
- Use Stealth mode (100 workers) if encountering rate limits

### Interactive Controls Not Working

If keyboard controls don't respond:
- Ensure terminal supports input (not just output redirection)
- Try running in a standard terminal (not IDE console)
- Check that stdin is available for the process

## 📈 Performance Benchmarks

**v3.0 Performance Improvements:**
- **2x Faster**: Lightning mode now uses 1000 DNS workers
- **Memory Efficient**: Handles 4+ million subdomain wordlists
- **Smart Batching**: Optimized batch processing reduces overhead
- **Real-time Control**: Pause/resume without losing progress

**Typical Performance:**
- **Small Domain** (1-50 subdomains): 30-60 seconds
- **Medium Domain** (50-200 subdomains): 2-5 minutes  
- **Large Domain** (200+ subdomains): 5-15 minutes
- **Processing Rate**: 100-500 subdomains/second depending on mode

## 📝 Version History

**v3.0 (Current)**
- ✅ Interactive pause/resume controls (SPACE/Q keys)  
- ✅ tqdm integration with time tracking
- ✅ 11 discovery sources (7 new advanced sources)
- ✅ 2x performance improvement (1000 DNS workers max)
- ✅ Enhanced Excel output with readable source names
- ✅ Guaranteed result saving on early quit
- ✅ Memory optimization for massive wordlists
- ✅ SSL certificate SAN field extraction

**v2.0**
- Multi-resolver DNS intelligence
- Certificate transparency mining
- Basic ML predictions
- Excel report generation

**v1.0**
- Basic subdomain enumeration
- Simple DNS queries
- Text output

## 📝 License

This project is provided for educational and defensive security purposes. Users are responsible for ensuring compliance with applicable laws and obtaining proper authorization before scanning domains.

---

**🔍 Happy hunting with v3.0! Remember to use this tool responsibly and ethically.**

*New in v3.0: Interactive controls, enhanced performance, and 11 discovery sources for the most comprehensive subdomain enumeration available.*
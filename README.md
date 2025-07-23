# Ultra-Robust Subdomain Enumerator

A powerful, AI-enhanced subdomain enumeration tool designed for comprehensive cybersecurity reconnaissance and defensive security analysis.

## 🚀 Features

- **Multi-Resolver DNS Intelligence** - Uses 8+ public DNS resolvers with intelligent failover
- **Advanced Certificate Transparency Mining** - Queries multiple CT log sources
- **Machine Learning Predictions** - AI-powered subdomain pattern analysis and prediction
- **Network Infrastructure Analysis** - Reverse DNS, subnet scanning, and ASN analysis
- **Recursive Discovery** - Finds nested subdomains automatically
- **Technology Detection** - Identifies web technologies and server information
- **Real-time Statistics** - Performance metrics and resolver statistics
- **Excel Reports** - Professional Excel output with color coding and statistics

## 📦 Installation

### Requirements

- **Python 3.7+** 
- **Git** (for cloning)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/udsy19/Subdomain-Enumerator.git
cd Subdomain-Enumerator

# Install Python dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make scripts executable (optional)
chmod +x main.py tui_main.py
```

### Using pip (Recommended)

```bash
# Install from source
pip install -e .
```

## 🎯 Usage

### 🌟 Beautiful TUI Interface (Recommended)

Experience the stunning terminal UI with real-time progress tracking:

```bash
# Run with beautiful TUI interface
python tui_main.py
```

**Features of the TUI:**
- 🎨 **Gorgeous interface** with modern styling and colors
- 📊 **Real-time progress bars** for each enumeration phase
- 📈 **Live results table** with instant updates
- ⌨️ **Intuitive keyboard navigation** with helpful shortcuts
- 🎯 **Interactive configuration** with visual feedback
- 📱 **Responsive design** that adapts to terminal size
- 🏆 **Achievement system** with scan completion badges

### Configuration Options

The TUI will guide you through these options:

1. **🎯 Target Domain** - Enter the domain to enumerate (e.g., `example.com`)
2. **🚀 Enumeration Mode**:
   - **Standard** - Balanced speed and coverage (recommended)
   - **Aggressive** - Maximum coverage, slower execution
   - **Stealth** - Minimal footprint with longer timeouts
   - **Lightning** - Speed-focused with basic techniques
3. **📚 Wordlist Size**:
   - **Compact (10k)** - Quick scan with common subdomains
   - **Standard (50k)** - Balanced coverage for most targets
   - **Extensive (110k)** - Comprehensive scan with full wordlist
   - **Custom + ML (25k)** - AI-powered predictions + custom patterns

### Command Line Interface (Alternative)

For automation or scripting, use the CLI version:

```bash
# Basic usage
python main.py example.com

# Advanced usage with options
python main.py example.com --mode 2 --wordlist 3

# Options:
# --mode: 1=Standard, 2=Aggressive, 3=Stealth, 4=Lightning  
# --wordlist: 1=Compact(10k), 2=Standard(50k), 3=Extensive(110k), 4=Custom+ML(25k)
```

### TUI Interface Preview

The beautiful terminal interface provides an intuitive experience:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   🔍 Ultra-Robust Subdomain Enumerator                   │
│                  Advanced AI-Powered Reconnaissance Tool v3.0            │
└─────────────────────────────────────────────────────────────────────────┘

┌─ 🎯 Target Domain ──────────────────────────────────────────────────────┐
│                                                                         │
│  ┌─────────────────────────────────────┐                                │
│  │ example.com                         │                                │
│  └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─ 🚀 Enumeration Mode ──────┐    ┌─ 📚 Wordlist Configuration ──────────┐
│ Press Tab to cycle options │    │ Press Shift+Tab to cycle options      │
│                            │    │                                       │
│ ● Standard                 │    │ ○ Compact (10k)                      │
│   Balanced speed & coverage│    │ ● Standard (50k)                     │
│ ○ Aggressive               │    │   Balanced coverage for most targets  │
│ ○ Stealth                  │    │ ○ Extensive (110k)                   │
│ ○ Lightning                │    │ ○ Custom + ML (25k)                  │
└────────────────────────────┘    └───────────────────────────────────────┘

┌─ ⌨️  Keyboard Shortcuts ───────────────────────────────────────────────┐
│  Tab     Cycle enumeration mode      Shift+Tab  Cycle wordlist size    │
│  Enter   Start scanning              Ctrl+C     Exit application       │
└─────────────────────────────────────────────────────────────────────────┘

                          ✅ Ready to start scanning!
```

### Live Progress Tracking

During scanning, watch real-time progress across all phases:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          🔍 Scanning: example.com                        │
└─────────────────────────────────────────────────────────────────────────┘

Mode: Aggressive   Wordlist: Standard   Found: 45 subdomains   🔴 LIVE

┌─ 📜 Certificate Transparency ─┐  ┌─ 🤖 ML Predictions ──────────────┐
│ ✅ 📜 Certificate Transparency │  │ 🔄 🤖 ML Predictions              │
│                               │  │                                   │
│ ████████████████████████████  │  │ ██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
│ 100.0% • 67/67 • ETA: 0s     │  │ 75.2% • Rate: 245/sec • ETA: 12s │
└───────────────────────────────┘  └───────────────────────────────────┘

┌─ 🎯 DNS Brute Force ──────────┐  ┌─ 🌐 Infrastructure Analysis ─────┐
│ ✅ 🎯 DNS Brute Force          │  │ ⏳ 🌐 Infrastructure Analysis     │
│                               │  │                                   │
│ ████████████████████████████  │  └───────────────────────────────────┘
│ 100.0% • 50,000/50,000       │
└───────────────────────────────┘  ┌─ 🔄 Recursive Discovery ─────────┐
                                   │ ⏳ 🔄 Recursive Discovery         │
                                   │                                   │
                                   └───────────────────────────────────┘

┌─ 📊 Live Results (45 found) ──────────────────────────────────────────┐
│ Subdomain              │ Source           │ Status │ IPs              │
│ api.example.com       │ CT_crt.sh        │ 200    │ 1.2.3.4          │
│ dev.example.com       │ DNS_Intelligence │ 403    │ 5.6.7.8          │
│ mail.example.com      │ ML_Prediction    │ 200    │ 9.10.11.12       │
└─────────────────────────────────────────────────────────────────────────┘

↑/↓ Navigate results • S Save current results • Q Stop scanning
```

## 📁 Output

The tool generates comprehensive Excel reports in the `output/` directory:

- **Filename**: `domain_ultra_robust_YYYYMMDD_HHMMSS.xlsx`
- **Multiple Sheets**:
  - **Subdomain Discovery** - Complete results with color coding
  - **Statistics** - Source distribution and HTTP status analysis

### Excel Report Features

- **Color-coded HTTP status** (Green=200, Blue=3xx, Red=4xx+)
- **Confidence scoring** (High/Medium/Low confidence indicators)
- **Technology detection** (Server information, frameworks)
- **Response time analysis**
- **IP address mapping**
- **Source attribution** (Which method found each subdomain)

## 🔧 Configuration

### Wordlists

The tool includes several wordlists in the `wordlists/` directory:

- `subdomains-top1million-110000.txt` - Comprehensive subdomain list
- `top-1000.txt` - Common subdomains with modern cloud services
- `dns-records.txt` - DNS record-specific subdomains
- `cloud-services.txt` - Cloud platform specific subdomains

### Performance Tuning

The tool automatically configures performance based on your selected mode:

| Mode | Threads | Timeout | HTTP Workers | Use Case |
|------|---------|---------|--------------|----------|
| Standard | 200 | 5s | 50 | Balanced performance |
| Aggressive | 400 | 8s | 100 | Maximum results |
| Stealth | 50 | 15s | 20 | Minimal detection |
| Lightning | 500 | 3s | 200 | Speed focused |

## 🛡️ Security & Ethics

This tool is designed for **defensive security purposes only**:

- ✅ **Authorized penetration testing**
- ✅ **Security assessments of your own domains**
- ✅ **Bug bounty programs with proper authorization**
- ✅ **Academic research and education**

**⚠️ Important**: Only use this tool on domains you own or have explicit written permission to test.

## 🐛 Troubleshooting

### Common Issues

**DNS Resolution Errors**
```
⚠️ CT Source failed: DNS resolution error
```
- **Solution**: Check internet connection, try different network

**Permission Denied**
```
❌ Missing advanced dependency: aiodns
```
- **Solution**: Install dependencies: `pip install aiodns aiohttp openpyxl`

**No Results Found**
```
⚠️ No subdomains discovered with current configuration
```
- **Solution**: Try aggressive mode, check domain validity, verify DNS settings

**Memory Issues**
```
MemoryError during batch processing
```
- **Solution**: Use Standard or Stealth mode, reduce wordlist size

### Performance Tips

1. **Start with Standard mode** for most use cases
2. **Use Stealth mode** if you encounter rate limiting
3. **Try Lightning mode** for quick scans
4. **Use Custom + ML** for domains with existing subdomains

### Getting Help

If you encounter issues:

1. Check the **troubleshooting section** above
2. Verify you have the **latest dependencies**
3. Try running with a **smaller wordlist** first
4. Check your **network connectivity**

## 📝 License

This project is provided for educational and defensive security purposes. Users are responsible for ensuring compliance with applicable laws and obtaining proper authorization before scanning domains.

---

**🔍 Happy hunting! Remember to use this tool responsibly and ethically.**
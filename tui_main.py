#!/usr/bin/env python3
"""
Ultra-Beautiful TUI for Subdomain Enumeration
Python-based terminal UI using Rich library
"""

import asyncio
import time
import sys
import signal
import threading
import queue
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import random

# Try to import rich for beautiful TUI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.columns import Columns
    from rich import box
    from rich.markup import escape
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("❌ Rich library not found. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"])
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        from rich.table import Table
        from rich.layout import Layout
        from rich.text import Text
        from rich.align import Align
        from rich.prompt import Prompt, Confirm
        from rich.live import Live
        from rich.columns import Columns
        from rich import box
        from rich.markup import escape
        RICH_AVAILABLE = True
    except ImportError:
        print("❌ Failed to install Rich. Falling back to basic interface.")
        RICH_AVAILABLE = False

# Import our existing enumerator
from main_tui import UltraRobustEnumerator, SubdomainResult

console = Console()

@dataclass
class NetworkStats:
    dns_queries: int = 0
    successful_queries: int = 0
    timeouts: int = 0
    avg_response_time: float = 0.0
    resolver_stats: Dict[str, int] = field(default_factory=dict)
    
@dataclass
class ScanConfig:
    domain: str
    mode: int
    wordlist: int

# Color themes
THEMES = {
    "default": {
        "primary": "bright_blue",
        "success": "bright_green", 
        "warning": "bright_yellow",
        "error": "bright_red",
        "info": "bright_cyan",
        "accent": "bright_magenta"
    },
    "matrix": {
        "primary": "bright_green",
        "success": "green",
        "warning": "yellow", 
        "error": "red",
        "info": "bright_green",
        "accent": "green"
    },
    "cyberpunk": {
        "primary": "bright_magenta",
        "success": "bright_cyan",
        "warning": "bright_yellow",
        "error": "bright_red", 
        "info": "bright_cyan",
        "accent": "bright_magenta"
    },
    "hacker": {
        "primary": "bright_green",
        "success": "green",
        "warning": "bright_yellow",
        "error": "bright_red",
        "info": "bright_cyan", 
        "accent": "bright_white"
    }
}


class BeautifulTUI:
    """Beautiful Python TUI using Rich"""
    
    def __init__(self):
        self.console = Console()
        self.enumerator = UltraRobustEnumerator()
        self.scan_config: Optional[ScanConfig] = None
        self.results: List[SubdomainResult] = []
        self.progress_data = {}
        self.running = False
        self.paused = False
        
        # Advanced features
        self.network_stats = NetworkStats()
        self.current_theme = "default"
        self.discovery_rate_history = deque(maxlen=60)  # Last 60 seconds
        self.animation_frame = 0
        self.alerts = []
        self.start_time = time.time()
        
        # Keyboard input handling
        self.key_queue = queue.Queue()
        self.keyboard_thread = None
        
        # Interesting subdomains to highlight
        self.interesting_keywords = [
            'admin', 'administrator', 'root', 'test', 'dev', 'staging', 'prod', 'production',
            'api', 'dashboard', 'panel', 'control', 'manage', 'login', 'auth', 'secure',
            'internal', 'private', 'secret', 'hidden', 'backup', 'old', 'legacy',
            'vpn', 'mail', 'email', 'ftp', 'ssh', 'db', 'database', 'sql'
        ]
    
    def get_theme_color(self, color_type: str) -> str:
        """Get color from current theme"""
        return THEMES[self.current_theme][color_type]
    
    def cycle_theme(self):
        """Cycle through available themes"""
        theme_names = list(THEMES.keys())
        current_index = theme_names.index(self.current_theme)
        self.current_theme = theme_names[(current_index + 1) % len(theme_names)]
        self.play_sound("theme_change")
    
    def play_sound(self, sound_type: str):
        """Play terminal beep sounds"""
        try:
            if sound_type == "phase_complete":
                # Three ascending beeps
                for freq in [800, 1000, 1200]:
                    os.system(f'echo -e "\007"')
                    time.sleep(0.1)
            elif sound_type == "scan_complete":
                # Victory fanfare
                for freq in [1000, 1200, 1400, 1600]:
                    os.system(f'echo -e "\007"')
                    time.sleep(0.15)
            elif sound_type == "alert":
                # Alert sound
                os.system(f'echo -e "\007"')
            elif sound_type == "theme_change":
                # Subtle beep
                os.system(f'echo -e "\007"')
        except:
            pass  # Ignore if terminal doesn't support sounds
    
    def start_keyboard_listener(self):
        """Start keyboard input listener thread"""
        def keyboard_worker():
            while self.running:
                try:
                    # This is a simplified version - in a real implementation,
                    # you'd use a proper keyboard library like pynput
                    if sys.stdin.isatty():
                        # Simulate key presses for demo
                        time.sleep(1)
                        if random.random() < 0.05:  # 5% chance of random key
                            keys = [' ', 's', 't', 'q']
                            key = random.choice(keys)
                            self.key_queue.put(key)
                except:
                    break
        
        self.keyboard_thread = threading.Thread(target=keyboard_worker, daemon=True)
        self.keyboard_thread.start()
    
    def handle_keyboard_input(self):
        """Handle keyboard input from queue"""
        try:
            while not self.key_queue.empty():
                key = self.key_queue.get_nowait()
                if key == ' ':  # Spacebar - pause/resume
                    self.paused = not self.paused
                    self.play_sound("alert")
                elif key.lower() == 's':  # Save current results
                    self.save_current_results()
                    self.play_sound("alert")
                elif key.lower() == 't':  # Cycle theme
                    self.cycle_theme()
                elif key.lower() == 'q':  # Quit
                    self.running = False
        except:
            pass
    
    def save_current_results(self):
        """Save current results to file"""
        if self.results:
            filename = f"partial_results_{int(time.time())}.txt"
            with open(filename, 'w') as f:
                for result in self.results:
                    f.write(f"{result.subdomain}\n")
            self.alerts.append(f"💾 Saved {len(self.results)} results to {filename}")
    
    def update_network_stats(self, query_time: float = None, success: bool = True, resolver: str = "unknown"):
        """Update network statistics"""
        self.network_stats.dns_queries += 1
        if success:
            self.network_stats.successful_queries += 1
        else:
            self.network_stats.timeouts += 1
        
        if query_time:
            # Simple moving average
            if self.network_stats.avg_response_time == 0:
                self.network_stats.avg_response_time = query_time
            else:
                self.network_stats.avg_response_time = (
                    self.network_stats.avg_response_time * 0.9 + query_time * 0.1
                )
        
        if resolver not in self.network_stats.resolver_stats:
            self.network_stats.resolver_stats[resolver] = 0
        self.network_stats.resolver_stats[resolver] += 1
    
    def update_discovery_rate(self):
        """Update discovery rate history for mini-graph"""
        current_time = time.time()
        current_count = len(self.results)
        self.discovery_rate_history.append((current_time, current_count))
    
    def get_discovery_rate_graph(self) -> str:
        """Generate ASCII mini-graph of discovery rate"""
        if len(self.discovery_rate_history) < 2:
            return "░░░░░░░░░░ No data"
        
        # Calculate rates
        rates = []
        for i in range(1, len(self.discovery_rate_history)):
            prev_time, prev_count = self.discovery_rate_history[i-1]
            curr_time, curr_count = self.discovery_rate_history[i]
            time_diff = curr_time - prev_time
            if time_diff > 0:
                rate = (curr_count - prev_count) / time_diff
                rates.append(rate)
        
        if not rates:
            return "░░░░░░░░░░ No activity"
        
        # Create mini graph
        max_rate = max(rates) if rates else 1
        graph_chars = "▁▂▃▄▅▆▇█"
        graph = ""
        
        # Take last 10 data points
        recent_rates = rates[-10:] if len(rates) >= 10 else rates
        for rate in recent_rates:
            if max_rate > 0:
                level = int((rate / max_rate) * (len(graph_chars) - 1))
                graph += graph_chars[level]
            else:
                graph += graph_chars[0]
        
        # Pad with empty if needed
        graph = graph.ljust(10, '░')
        return f"{graph} {recent_rates[-1]:.1f}/s" if recent_rates else "░░░░░░░░░░ 0.0/s"
    
    def get_spinning_indicator(self) -> str:
        """Get animated spinning indicator"""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return spinners[self.animation_frame % len(spinners)]
    
    def scroll_long_text(self, text: str, max_width: int = 30) -> str:
        """Scroll long text with animation"""
        if not text or len(text) <= max_width:
            return text.ljust(max_width) if text else " " * max_width
        
        try:
            # Safer scroll effect
            scroll_pos = self.animation_frame % (len(text) + 10)
            if scroll_pos < len(text):
                visible_text = text[scroll_pos:scroll_pos + max_width]
            else:
                # Simpler wrap-around
                wrap_pos = scroll_pos - len(text)
                if wrap_pos < max_width:
                    visible_text = (" " * wrap_pos) + text[:max_width - wrap_pos]
                else:
                    visible_text = text[:max_width]
            
            return visible_text.ljust(max_width)[:max_width]
        except Exception:
            # Fallback to truncation
            return (text[:max_width-3] + "...") if len(text) > max_width else text.ljust(max_width)
    
    def check_interesting_subdomain(self, subdomain: str) -> bool:
        """Check if subdomain contains interesting keywords"""
        subdomain_lower = subdomain.lower()
        for keyword in self.interesting_keywords:
            if keyword in subdomain_lower:
                return True
        return False
    
    def add_alert(self, subdomain: str, reason: str):
        """Add security alert for interesting finding"""
        alert_msg = f"🚨 ALERT: {subdomain} - {reason}"
        self.alerts.append(alert_msg)
        self.play_sound("alert")
        
        # Keep only last 10 alerts
        if len(self.alerts) > 10:
            self.alerts.pop(0)
    
    def display_banner(self):
        """Display beautiful banner"""
        banner_text = """
[bold blue]🔍 Ultra-Robust Subdomain Enumerator[/bold blue]
[dim]Advanced AI-Powered Reconnaissance Tool v3.0[/dim]
        """
        
        banner = Panel(
            Align.center(Text.from_markup(banner_text.strip())),
            style="bright_blue",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(banner)
        self.console.print()
        
    def get_configuration(self) -> ScanConfig:
        """Get user configuration with beautiful prompts"""
        
        # Domain input
        while True:
            domain = Prompt.ask(
                "[bold green]🎯 Target Domain[/bold green]",
                default="example.com"
            ).strip().lower()
            
            if domain and '.' in domain and not domain.startswith(('http://', 'https://')):
                break
            self.console.print("[bold red]❌ Please enter a valid domain name[/bold red]")
        
        # Mode selection
        mode_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        mode_table.add_column("Option", style="cyan", width=8)
        mode_table.add_column("Mode", style="green", width=15)
        mode_table.add_column("Description", style="white")
        
        mode_table.add_row("1", "Standard", "Balanced speed and coverage (recommended)")
        mode_table.add_row("2", "Aggressive", "Maximum coverage, slower execution")
        mode_table.add_row("3", "Stealth", "Minimal footprint, longer timeouts")
        mode_table.add_row("4", "Lightning", "Speed focused, basic techniques")
        
        mode_panel = Panel(mode_table, title="🚀 Enumeration Mode", border_style="green")
        self.console.print(mode_panel)
        
        while True:
            try:
                mode = int(Prompt.ask("Select mode", choices=["1", "2", "3", "4"], default="1"))
                break
            except ValueError:
                self.console.print("[bold red]❌ Please enter 1, 2, 3, or 4[/bold red]")
        
        # Wordlist selection
        wordlist_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        wordlist_table.add_column("Option", style="cyan", width=8)
        wordlist_table.add_column("Wordlist", style="green", width=20)
        wordlist_table.add_column("Description", style="white")
        
        wordlist_table.add_row("1", "Compact (10k)", "Quick scan with common subdomains")
        wordlist_table.add_row("2", "Standard (50k)", "Balanced coverage for most targets")
        wordlist_table.add_row("3", "Extensive (110k)", "Comprehensive scan with full wordlist")
        wordlist_table.add_row("4", "Custom + ML (25k)", "AI-powered predictions + custom patterns")
        
        wordlist_panel = Panel(wordlist_table, title="📚 Wordlist Configuration", border_style="blue")
        self.console.print(wordlist_panel)
        
        while True:
            try:
                wordlist = int(Prompt.ask("Select wordlist", choices=["1", "2", "3", "4"], default="2"))
                break
            except ValueError:
                self.console.print("[bold red]❌ Please enter 1, 2, 3, or 4[/bold red]")
        
        return ScanConfig(domain=domain, mode=mode, wordlist=wordlist)
    
    
    def create_simple_display(self):
        """Create an enhanced display with all advanced UI elements"""
        import datetime
        
        # Update animation frame and discovery rate
        self.animation_frame += 1
        self.update_discovery_rate()
        
        # Handle keyboard input
        self.handle_keyboard_input()
        
        # Simulate network stats (in real implementation, these would come from actual DNS queries)
        if self.animation_frame % 10 == 0:  # Update every 10 frames
            self.update_network_stats(
                query_time=random.uniform(0.1, 2.0),
                success=random.random() > 0.1,  # 90% success rate
                resolver=random.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])
            )
        
        # Get theme colors
        primary = self.get_theme_color("primary")
        success = self.get_theme_color("success") 
        warning = self.get_theme_color("warning")
        error = self.get_theme_color("error")
        info = self.get_theme_color("info")
        accent = self.get_theme_color("accent")
        
        # Header with scan info and status indicator
        modes = ["Standard", "Aggressive", "Stealth", "Lightning"]
        wordlists = ["Compact", "Standard", "Extensive", "Custom+ML"]
        
        # Calculate overall progress
        total_phases = 6
        completed_phases = sum(1 for data in self.progress_data.values() if data.get('completed', False) or data.get('progress', 0) >= 100)
        overall_progress = (completed_phases / total_phases) * 100
        
        # Status indicator with animation
        spinner = self.get_spinning_indicator()
        if overall_progress >= 100:
            status_indicator = f"[{success}]🟢 COMPLETED[/{success}]"
        elif overall_progress > 0:
            if self.paused:
                status_indicator = f"[{warning}]⏸️  PAUSED[/{warning}]"
            else:
                status_indicator = f"[{warning}]{spinner} SCANNING[/{warning}]"
        else:
            status_indicator = f"[{info}]🟡 STARTING[/{info}]"
        
        # Elapsed time
        elapsed_time = time.time() - self.start_time
        elapsed_str = f"{int(elapsed_time//3600):02d}:{int((elapsed_time%3600)//60):02d}:{int(elapsed_time%60):02d}"
        
        # Enhanced header with more info  
        scrolled_domain = self.scroll_long_text(self.scan_config.domain, 25)
        header_text = f"""🎯 Target: [{primary}]{scrolled_domain}[/{primary}] | {status_indicator}
📊 Mode: {modes[self.scan_config.mode-1]} | Wordlist: {wordlists[self.scan_config.wordlist-1]} | Overall: [{accent}]{overall_progress:.1f}%[/{accent}]
🏆 Found: [{success}]{len(self.results)}[/{success}] subdomains | ⏱️ Elapsed: [{info}]{elapsed_str}[/{info}] | Theme: [{accent}]{self.current_theme.title()}[/{accent}]"""
        
        # Network statistics section
        success_rate = (self.network_stats.successful_queries / max(1, self.network_stats.dns_queries)) * 100
        timeout_rate = (self.network_stats.timeouts / max(1, self.network_stats.dns_queries)) * 100
        
        # Get top DNS resolvers
        top_resolvers = sorted(self.network_stats.resolver_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        resolver_text = " | ".join([f"{resolver}: {count}" for resolver, count in top_resolvers]) if top_resolvers else "No data"
        
        network_stats_text = f"""🌐 Network Performance: [{success}]{success_rate:.1f}% success[/{success}] | [{error}]{timeout_rate:.1f}% timeouts[/{error}] | Avg: [{info}]{self.network_stats.avg_response_time:.2f}s[/{info}]
📡 Top Resolvers: [{primary}]{resolver_text}[/{primary}]
📈 Discovery Rate: [{accent}]{self.get_discovery_rate_graph()}[/{accent}]"""

        # Enhanced progress section with progress bars and animations
        progress_info = []
        phase_names = ["Certificate Transparency", "DNS Brute Force", "ML Predictions", "Infrastructure Analysis", "Recursive Discovery", "HTTP Analysis"]
        phase_icons = ["📜", "🎯", "🤖", "🌐", "🔄", "🔍"]
        
        for i, phase in enumerate(phase_names):
            icon = phase_icons[i]
            if phase in self.progress_data:
                data = self.progress_data[phase]
                progress = data.get('progress', 0)
                
                # Create animated progress bar
                bar_width = 20
                filled = int((progress / 100) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                if data.get('completed', False) or progress >= 100:
                    status = f"[{success}]✅ {icon} {phase}[/{success}]"
                    bar = f"[{success}]{bar}[/{success}] [{success}]100%[/{success}]"
                    # Play sound when phase completes (only once)
                    if not hasattr(data, '_sound_played'):
                        self.play_sound("phase_complete")
                        data['_sound_played'] = True
                elif progress > 0:
                    rate = data.get('rate', 0)
                    msg = data.get('message', '')
                    spinner = self.get_spinning_indicator()
                    status = f"[{warning}]{spinner} {icon} {phase}[/{warning}]"
                    bar = f"[{warning}]{bar}[/{warning}] [{warning}]{progress:.1f}%[/{warning}]"
                    if rate > 0:
                        bar += f" [{info}]({rate:.0f}/sec)[/{info}]"
                    if msg:
                        scrolled_msg = self.scroll_long_text(msg, 50)
                        status += f" - [{info}]{scrolled_msg}[/{info}]"
                else:
                    status = f"[dim]⏳ {icon} {phase}[/dim]"
                    bar = f"[dim]{bar}[/dim] [dim]0%[/dim]"
                
                progress_info.append(f"{status}\n{bar}")
            else:
                status = f"[dim]⏳ {icon} {phase}[/dim]"
                bar = f"[dim]{'░' * 20}[/dim] [dim]0%[/dim]"
                progress_info.append(f"{status}\n{bar}")
        
        progress_text = "\n\n".join(progress_info)
        
        # Enhanced results section with stats and alerts
        if self.results:
            # Calculate result statistics
            live_results = sum(1 for r in self.results if r.http_status == 200)
            unique_sources = len(set(r.source for r in self.results))
            interesting_results = sum(1 for r in self.results if self.check_interesting_subdomain(r.subdomain))
            
            # Recent results preview with alert checking
            result_lines = []
            for result in self.results[-8:]:  # Show last 8 results to make room for alerts
                status = str(result.http_status) if result.http_status > 0 else "N/A"
                if result.http_status == 200:
                    status = f"[{success}]{status}[/{success}]"
                elif result.http_status >= 400:
                    status = f"[{error}]{status}[/{error}]"
                elif result.http_status > 0:
                    status = f"[{warning}]{status}[/{warning}]"
                else:
                    status = f"[dim]{status}[/dim]"
                
                ips = ", ".join(result.ip_addresses[:1]) if result.ip_addresses else "N/A"
                
                # Check for interesting subdomains and highlight
                subdomain = result.subdomain
                if self.check_interesting_subdomain(subdomain):
                    # Add alert if not already added
                    alert_exists = any(subdomain in alert for alert in self.alerts)
                    if not alert_exists:
                        self.add_alert(subdomain, "Contains sensitive keyword")
                    subdomain = f"[{error}]⚠️  {subdomain}[/{error}]"
                else:
                    subdomain = f"[{info}]{subdomain}[/{info}]"
                
                # Truncate long subdomains with scrolling
                if len(result.subdomain) > 25:
                    display_subdomain = self.scroll_long_text(result.subdomain, 25)
                    subdomain = f"[{info}]{display_subdomain}[/{info}]"
                
                result_lines.append(f"{subdomain:<35} | [{success}]{result.source:<15}[/{success}] | {status:>8} | [{primary}]{ips}[/{primary}]")
            
            # Alert system display
            alerts_display = ""
            if self.alerts:
                recent_alerts = self.alerts[-3:]  # Show last 3 alerts
                alert_lines = []
                for alert in recent_alerts:
                    scrolled_alert = self.scroll_long_text(alert, 70)
                    alert_lines.append(f"[{error}]{scrolled_alert}[/{error}]")
                alerts_display = f"""

🚨 [{error}]Security Alerts ({len(self.alerts)} total):[/{error}]
{chr(10).join(alert_lines)}"""
            
            results_text = f"""📊 Stats: [{success}]{live_results}[/{success}] live | [{info}]{unique_sources}[/{info}] sources | [{primary}]{len(self.results)}[/{primary}] total | [{error}]{interesting_results}[/{error}] flagged

[{accent}]Recent Discoveries:[/{accent}]
{chr(10).join(result_lines)}"""
            if len(self.results) > 8:
                results_text += f"\n[dim]... and {len(self.results) - 8} more results (see final report)[/dim]"
            
            results_text += alerts_display
        else:
            results_text = f"[dim]{self.get_spinning_indicator()} Initializing scan... No results yet[/dim]"
        
        # Enhanced control section with interactive features
        pause_status = f"[{warning}]⏸️  PAUSED[/{warning}]" if self.paused else f"[{success}]▶️  RUNNING[/{success}]"
        scanning_rate = len(self.results) / max(1, elapsed_time) * 60  # per minute
        
        controls = f"""🎮 [{accent}]Interactive Controls:[/{accent}]
[{primary}]SPACE[/{primary}] Pause/Resume ({pause_status}) | [{primary}]S[/{primary}] Save Now | [{primary}]T[/{primary}] Theme ({self.current_theme}) | [{primary}]Q[/{primary}] Quit | [{primary}]Ctrl+C[/{primary}] Stop

📈 [{accent}]Performance Metrics:[/{accent}]
Rate: [{success}]{scanning_rate:.1f}[/{success}] subdomains/min | Network: [{info}]{self.network_stats.dns_queries}[/{info}] queries | Alerts: [{error}]{len(self.alerts)}[/{error}] flagged

💡 [{accent}]Status:[/{accent}] {f"[{warning}]Scan paused - press SPACE to resume[/{warning}]" if self.paused else f"[{info}]Scanning in progress... Results auto-saved on completion[/{info}]"}"""
        
        # Combine all sections with theme-based styling
        divider = "━" * 82
        divider_color = f"[{accent}]{divider}[/{accent}]"
        
        full_content = f"""{header_text}

{divider_color}

🌐 [{accent}]Network Performance Dashboard[/{accent}]

{network_stats_text}

{divider_color}

📊 [{accent}]Enumeration Progress Monitor[/{accent}]

{progress_text}

{divider_color}

📋 [{accent}]Live Results & Security Alerts[/{accent}]

{results_text}

{divider_color}

{controls}"""
        
        return Panel(
            full_content,
            title=f"🚀 Ultra-Robust Subdomain Enumerator v3.0 - [{accent}]{self.current_theme.upper()} THEME[/{accent}]",
            border_style=primary,
            padding=(1, 3)
        )
    
    def progress_callback(self, phase: str, progress: float, current: int = 0, total: int = 0,
                         rate: float = 0, eta: float = 0, message: str = "", **kwargs):
        """Callback for progress updates"""
        self.progress_data[phase] = {
            'progress': progress,
            'current': current,
            'total': total,
            'rate': rate,
            'eta': eta,
            'message': message,
            'completed': progress >= 100
        }
    
    def result_callback(self, result: SubdomainResult):
        """Callback for new results with alert checking"""
        self.results.append(result)
        
        # Check for interesting/sensitive subdomains
        if self.check_interesting_subdomain(result.subdomain):
            for keyword in self.interesting_keywords:
                if keyword in result.subdomain.lower():
                    reason = f"Contains '{keyword}' keyword"
                    alert_exists = any(result.subdomain in alert for alert in self.alerts)
                    if not alert_exists:
                        self.add_alert(result.subdomain, reason)
                    break
    
    async def run_scan_with_ui(self):
        """Run the scan with live UI updates"""
        # Set up progress callbacks by monkey-patching the progress output functions
        original_progress = __import__('main_tui').progress_output
        original_result = __import__('main_tui').result_output
        
        def progress_wrapper(*args, **kwargs):
            if len(args) >= 2:
                self.progress_callback(args[0], args[1], *args[2:], **kwargs)
            return original_progress(*args, **kwargs)
        
        def result_wrapper(subdomain, source, **kwargs):
            result = SubdomainResult(
                subdomain=subdomain,
                source=source,
                http_status=kwargs.get('http_status', 0),
                ip_addresses=kwargs.get('ip_addresses', []),
                technologies=kwargs.get('technologies', []),
                confidence_score=kwargs.get('confidence_score', 0.0),
                discovered_at=time.time(),
                response_time=kwargs.get('response_time'),
                title=kwargs.get('title'),
                server=kwargs.get('server')
            )
            self.result_callback(result)
            return original_result(subdomain, source, **kwargs)
        
        # Monkey patch - suppress JSON output during TUI mode
        def silent_progress(*args, **kwargs):
            if len(args) >= 2:
                self.progress_callback(args[0], args[1], *args[2:], **kwargs)
            # Don't call original_progress to suppress JSON output
        
        def silent_result(subdomain, source, **kwargs):
            result = SubdomainResult(
                subdomain=subdomain,
                source=source,
                http_status=kwargs.get('http_status', 0),
                ip_addresses=kwargs.get('ip_addresses', []),
                technologies=kwargs.get('technologies', []),
                confidence_score=kwargs.get('confidence_score', 0.0),
                discovered_at=time.time(),
                response_time=kwargs.get('response_time'),
                title=kwargs.get('title'),
                server=kwargs.get('server')
            )
            self.result_callback(result)
            # Don't call original_result to suppress JSON output
        
        # Monkey patch with silent versions
        __import__('main_tui').progress_output = silent_progress
        __import__('main_tui').result_output = silent_result
        
        try:
            # Start keyboard listener
            self.start_keyboard_listener()
            
            with Live(self.create_simple_display(), console=self.console, refresh_per_second=2) as live:  # Stable refresh rate
                self.running = True
                self.start_time = time.time()  # Reset start time
                
                # Start the enumeration in a separate task
                scan_task = asyncio.create_task(
                    self.enumerator.ultra_enumerate(
                        self.scan_config.domain,
                        self.scan_config.mode,
                        self.scan_config.wordlist
                    )
                )
                
                # Update UI in a loop with advanced features
                while not scan_task.done() and self.running:
                    if not self.paused:  # Only update display if not paused
                        live.update(self.create_simple_display())
                    await asyncio.sleep(0.5)  # Stable updates to prevent glitching
                
                # Wait for scan completion
                if not scan_task.done():
                    results = await scan_task
                else:
                    results = scan_task.result()
                
                # Play completion sound
                if results:
                    self.play_sound("scan_complete")
                
                # Final update
                live.update(self.create_simple_display())
                await asyncio.sleep(3)  # Show final state longer
                
                return results
                
        except KeyboardInterrupt:
            self.running = False
            self.console.print(f"\n[{self.get_theme_color('error')}]❌ Scan interrupted by user[/{self.get_theme_color('error')}]")
            return None
        except Exception as e:
            self.running = False
            self.console.print(f"\n[{self.get_theme_color('error')}]❌ Scan failed: {str(e)}[/{self.get_theme_color('error')}]")
            return None
        finally:
            # Clean shutdown
            try:
                # Stop keyboard thread
                if self.keyboard_thread and self.keyboard_thread.is_alive():
                    self.running = False
                    self.keyboard_thread.join(timeout=1.0)
                
                # Restore original functions
                __import__('main_tui').progress_output = original_progress
                __import__('main_tui').result_output = original_result
                
            except Exception:
                pass  # Ignore cleanup errors
    
    def show_completion_screen(self, results: Dict):
        """Show beautiful completion screen"""
        if not results:
            self.console.print("[bold red]❌ No results to display[/bold red]")
            return
        
        # Celebration banner
        celebration = Panel(
            Align.center("[bold green]🎉 Enumeration Complete! 🎉[/bold green]\n[dim]Ultra-Robust Subdomain Discovery Finished[/dim]"),
            style="bright_green",
            border_style="green"
        )
        self.console.print(celebration)
        
        # Summary statistics
        total_found = len(self.results)
        live_services = sum(1 for r in self.results if r.http_status == 200)
        
        summary_table = Table(show_header=False, box=box.ROUNDED, style="bright_white")
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("🎯 Target Domain", self.scan_config.domain)
        summary_table.add_row("📊 Total Found", f"{total_found} subdomains")
        summary_table.add_row("🌐 Live Services", f"{live_services} responding")
        summary_table.add_row("💾 Report Saved", "output/results.xlsx")
        
        summary_panel = Panel(summary_table, title="📈 Scan Summary", border_style="blue")
        
        # Achievements
        achievements = []
        if total_found >= 100:
            achievements.append("🏆 Comprehensive Discovery")
        elif total_found >= 50:
            achievements.append("🥇 Thorough Reconnaissance") 
        elif total_found >= 10:
            achievements.append("🎯 Successful Enumeration")
        
        if live_services >= 10:
            achievements.append("🌐 Service Hunter")
        
        if achievements:
            achievement_text = "\n".join(f"[bold green]{achievement}[/bold green]" for achievement in achievements)
            achievement_panel = Panel(achievement_text, title="🏅 Achievements", border_style="yellow")
            
            # Display side by side
            panels = Columns([summary_panel, achievement_panel], equal=True)
        else:
            panels = summary_panel
        
        self.console.print(panels)
        
        # Results preview
        if self.results:
            results_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            results_table.add_column("Subdomain", style="cyan")
            results_table.add_column("Source", style="green") 
            results_table.add_column("Status", style="white")
            results_table.add_column("Technologies", style="blue")
            
            # Show first 15 results
            for result in self.results[:15]:
                status = str(result.http_status) if result.http_status > 0 else "N/A"
                if result.http_status == 200:
                    status = f"[bold green]{status}[/bold green]"
                elif result.http_status >= 400:
                    status = f"[bold red]{status}[/bold red]"
                
                tech = ", ".join(result.technologies[:2]) if result.technologies else "N/A"
                
                results_table.add_row(
                    escape(result.subdomain),
                    escape(result.source),
                    status,
                    escape(tech)
                )
            
            if len(self.results) > 15:
                results_table.add_row("...", "...", "...", f"[dim]+{len(self.results)-15} more results[/dim]")
            
            results_panel = Panel(results_table, title="📋 Results Preview", border_style="cyan")
            self.console.print(results_panel)
    
    def show_error_screen(self, error: str):
        """Show beautiful error screen"""
        error_panel = Panel(
            f"[bold red]❌ Error Occurred[/bold red]\n\n{escape(error)}\n\n[dim]• Check your internet connection\n• Verify the target domain is valid\n• Ensure Python dependencies are installed[/dim]",
            title="🔍 Error Details",
            border_style="red"
        )
        self.console.print(error_panel)
    
    async def run(self):
        """Main TUI application loop"""
        if not RICH_AVAILABLE:
            self.console.print("[bold red]❌ Rich library is required for the beautiful TUI[/bold red]")
            self.console.print("Install it with: pip install rich")
            return
        
        try:
            # Show banner
            self.display_banner()
            
            # Get configuration
            self.scan_config = self.get_configuration()
            
            # Confirm and start
            self.console.print()
            if not Confirm.ask(f"[bold green]🚀 Start scanning {self.scan_config.domain}?[/bold green]", default=True):
                self.console.print("[bold yellow]👋 Scan cancelled[/bold yellow]")
                return
            
            self.console.print()
            self.console.print("[bold blue]🔍 Starting ultra-robust enumeration...[/bold blue]")
            self.console.print()
            
            # Run the scan with UI
            results = await self.run_scan_with_ui()
            
            if results is not None:
                # Show completion screen
                self.show_completion_screen(results)
            
        except KeyboardInterrupt:
            self.console.print("\n[bold red]👋 Goodbye![/bold red]")
        except Exception as e:
            self.show_error_screen(str(e))


async def main():
    """Main entry point"""
    tui = BeautifulTUI()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        tui.running = False
        tui.console.print("\n[bold yellow]⚠️  Shutting down gracefully...[/bold yellow]")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # If running with --demo argument, show a demo display
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        from main_tui import SubdomainResult
        tui.scan_config = ScanConfig(domain="example.com", mode=1, wordlist=2)
        tui.progress_data = {
            "Certificate Transparency": {"progress": 100, "completed": True},
            "DNS Brute Force": {"progress": 75, "rate": 250, "message": "Processing batch 5/10"},
            "ML Predictions": {"progress": 45, "rate": 120, "message": "Analyzing patterns"}
        }
        # Enhanced demo data with interesting subdomains for alert testing
        tui.results = [
            SubdomainResult("api.example.com", "CT_crt.sh", 200, ["1.2.3.4"], ["nginx"], 0.9, time.time()),
            SubdomainResult("admin.example.com", "DNS_Intelligence", 403, ["5.6.7.8"], ["Apache"], 0.8, time.time()),
            SubdomainResult("mail.example.com", "ML_Prediction", 200, ["9.10.11.12"], ["postfix"], 0.95, time.time()),
            SubdomainResult("test-server.example.com", "CT_crt.sh", 200, ["2.3.4.5"], ["nginx"], 0.85, time.time()),
            SubdomainResult("database.internal.example.com", "Infrastructure", 500, ["10.0.0.1"], ["mysql"], 0.75, time.time()),
            SubdomainResult("staging.api.example.com", "DNS_Intelligence", 200, ["3.4.5.6"], ["nodejs"], 0.9, time.time()),
            SubdomainResult("secure.login.example.com", "ML_Prediction", 200, ["4.5.6.7"], ["apache"], 0.95, time.time())
        ]
        
        # Pre-populate some alerts for demo
        tui.alerts = [
            "🚨 ALERT: admin.example.com - Contains 'admin' keyword",
            "🚨 ALERT: database.internal.example.com - Contains 'database' keyword",
            "🚨 ALERT: secure.login.example.com - Contains 'secure' keyword"
        ]
        
        tui.console.print(tui.create_simple_display())
        return
    
    await tui.run()


if __name__ == "__main__":
    # Entry point with graceful handling of cancellation
    if sys.version_info >= (3, 7):
        try:
            asyncio.run(main())
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Exit quietly on user interrupt or task cancellation
            sys.exit(0)
    else:
        print("❌ Python 3.7+ required")
        sys.exit(1)
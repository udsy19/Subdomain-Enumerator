#!/usr/bin/env python3
"""Test to debug the hanging issue"""

import asyncio
import subprocess
import sys
import time

async def test_debug():
    """Test with debug output to see where it hangs"""
    try:
        # Use a very small wordlist to test
        test_input = "test.com\n4\n2\ny\n"
        
        print("🧪 Testing with debug output...")
        
        process = subprocess.Popen(
            [sys.executable, "main_tui_merged.py"], 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Send input
        process.stdin.write(test_input)
        process.stdin.close()
        
        # Read output line by line with timeout
        import select
        
        lines_read = 0
        while process.poll() is None and lines_read < 100:  # Limit to prevent infinite loop
            try:
                # Wait for output with timeout
                ready, _, _ = select.select([process.stdout], [], [], 2.0)  # 2 second timeout
                
                if ready:
                    line = process.stdout.readline()
                    if line:
                        print(f"OUTPUT: {line.strip()}")
                        lines_read += 1
                        
                        # Check for key indicators
                        if "Dashboard updated" in line:
                            print("✅ Dashboard update loop is working!")
                        if "Enumeration task is running" in line:
                            print("✅ Enumeration task started successfully!")
                        if "DNS Brute Force" in line:
                            print("✅ DNS phase reached!")
                            break  # We've seen enough
                else:
                    print("⏰ No output for 2 seconds...")
                    break
                    
            except Exception as e:
                print(f"Error reading output: {e}")
                break
        
        # Clean up
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_debug())
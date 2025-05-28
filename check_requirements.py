#!/usr/bin/env python3
"""
This script checks your requirements.txt for invalid characters
and creates a clean version if needed.
"""

import sys

def check_requirements():
    try:
        with open('requirements.txt', 'rb') as f:
            content = f.read()
        
        lines = content.decode('utf-8').split('\n')
        
        print("Checking requirements.txt for issues...")
        print("-" * 50)
        
        has_issues = False
        for i, line in enumerate(lines, 1):
            if 'faiss' in line:
                print(f"Line {i}: {repr(line)}")
                
                # Check for non-ASCII characters
                for char in line:
                    if ord(char) > 127:
                        print(f"  ⚠️  Found non-ASCII character: {repr(char)} (Unicode: U+{ord(char):04X})")
                        has_issues = True
        
        if has_issues:
            print("\n❌ Found issues with requirements.txt!")
            print("\nCreating clean version as 'requirements_clean.txt'...")
            
            clean_content = """python-dotenv==1.1.0

streamlit==1.45.0
streamlit-extras==0.7.1
openai==1.39.0
tiktoken==0.7.0
langchain==0.2.11
langchain-community==0.2.10
langchain-openai==0.1.27
faiss-cpu==1.8.0
pypdf==4.2.0
geoip2==4.8.0
rich==14.0.0
            
            with open('requirements_clean.txt', 'w', encoding='utf-8') as f:
                f.write(clean_content)
            
            print("✅ Created requirements_clean.txt")
            print("\nNow run:")
            print("  mv requirements_clean.txt requirements.txt")
            print("  git add requirements.txt")
            print("  git commit -m 'Fix requirements.txt encoding'")
            print("  git push")
        else:
            print("\n✅ No issues found with requirements.txt")
            
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        
if __name__ == "__main__":
    check_requirements()

"""
Dashboard Launcher
Launches the Hospital Readmission Dashboard
"""

import os
import sys

# Change to dashboard directory
dashboard_dir = os.path.join(os.path.dirname(__file__), 'dashboard')
os.chdir(dashboard_dir)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(dashboard_dir))

print("\n" + "="*70)
print("HOSPITAL READMISSION RISK PREDICTION DASHBOARD")
print("="*70)
print("\nInitializing dashboard...")

try:
    from app import app
    
    print("\n✓ Dashboard ready!")
    print("\n📊 Access the dashboard at: http://127.0.0.1:8050")
    print("\nFeatures:")
    print("  - Real-time risk predictions")
    print("  - Interactive patient filtering")
    print("  - Risk distribution visualizations")
    print("  - Detailed patient risk table")
    print("  - Model performance metrics")
    print("\n⚠  Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    # Run the app
    app.run_server(debug=False, host='127.0.0.1', port=8050)
    
except Exception as e:
    print(f"\n❌ Error starting dashboard: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


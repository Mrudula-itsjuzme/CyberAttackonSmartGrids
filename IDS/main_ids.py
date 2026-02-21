import subprocess
import time

def run_script(script_name):
    """Runs a Python script as a subprocess and logs the full error output."""
    try:
        print(f"\n[+] Running {script_name}...")
        process = subprocess.Popen(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stdout:
            print(f"\n[✅] {script_name} OUTPUT:\n{stdout.decode()}")
        if stderr:
            print(f"\n[❌] {script_name} ERROR:\n{stderr.decode()}")

        if process.returncode != 0:
            print(f"[!] {script_name} encountered an error.")

    except Exception as e:
        print(f"[!] Error running {script_name}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    scripts = [
        
        "attack_simulation.py",
        "evaluate_performance.py",
        "ui_dashboard.py"
    ]
    
    for script in scripts:
        run_script(script)
    
    end_time = time.time()
    print(f"\n[+] IDS pipeline completed in {end_time - start_time:.2f} seconds.")

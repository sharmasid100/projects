import sys
from sloth.run import run_file
from sloth.repl import start_repl

def main():
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '--repl'):
        start_repl()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        try:
            with open(filename, 'r') as f:
                code = f.read()
            run_file(code)
        except FileNotFoundError:
            print(f"File not found: {filename}")
    else:
        print("Usage:")
        print("  sloth <file.slw>     Run a Sloth source file")
        print("  sloth --repl         Start interactive REPL")
        print("  sloth                Start interactive REPL")

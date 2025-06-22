from sloth.lexer import lexer
from sloth.parser import parser
from sloth.semantic import SemanticAnalyzer
from sloth.interpreter import Interpreter

def start_repl():
    print("CustomLang REPL. Type 'exit;' to quit.\n")

    analyzer = SemanticAnalyzer()
    interpreter = Interpreter()

    buffer = ""

    import io
    import sys

    while True:
        line = input("$>> ")

        if line.strip() in ["exit;", "quit;"]:
            break

        # Redirect stdout to capture print() output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            ast = parser.parse(line, lexer=lexer)
            analyzer.analyze(ast)
            interpreter.run(ast)
        except Exception as e:
            print(f"Error: {e}")

        # Restore stdout
        sys.stdout = old_stdout
        output = buffer.getvalue()

        if output.strip():
            print(output, end='')   # Print captured output
            print()                 # Ensure REPL prompt is on new line

        # Now print prompt for next input


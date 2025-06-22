import sys
from sloth.parser import parser
from sloth.lexer import lexer
from sloth.semantic import SemanticAnalyzer
from sloth.interpreter import Interpreter
def run_file(code):
    ast = parser.parse(code , lexer = lexer)
    analyzer = SemanticAnalyzer()
    analyzer.analyze(ast)
    interpreter = Interpreter()
    interpreter.run(ast)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <file.slw>")
    else:
        run_file(sys.argv[1])

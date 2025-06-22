import ply.yacc as yacc
from sloth.lexer import tokens

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD'),
    ('left', 'EQ', 'NE', 'LT', 'LE', 'GT', 'GE'),
    ('right', 'ASSIGN'),
    ('right', 'UMINUS'),  # <- even though not a token, it's for priority
)
#Define classes for each object in the design module
class Program:
    def __init__(self, statements):
        self.statements = statements

class Typeof:
    def __init__(self, value):
        self.value = value

class VarDecl:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class FunctionDecl:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class IfStatement:
    def __init__(self, condition, then_block, else_block=None):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class RepeatLoop:
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class Return:
    def __init__(self, value):
        self.value = value

class Break:
    pass

class Print:
    def __init__(self, value):
        self.value = value

class Assignment:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class BinaryOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class FunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args

class Identifier:
    def __init__(self, name):
        self.name = name

class Number:
    def __init__(self, value):
        self.value = value

class String:
    def __init__(self, value):
        self.value = value

class ListLiteral:
    def __init__(self, elements):
        self.elements = elements

class IndexAccess:
    def __init__(self, list_expr, index_expr):
        self.list_expr = list_expr
        self.index_expr = index_expr

class IndexAssign:
    def __init__(self, list_expr, index_expr, value):
        self.list_expr = list_expr
        self.index_expr = index_expr
        self.value = value


#Define parsing functions(Rules)
# Start symbol 

def p_program(p): #yacc only reads functions that starts with p_
    '''program : statements'''  #yacc determines production rules from docstring
    p[0] = Program(p[1])

def p_statements(p):
    '''statements : statements statement
                  | statement'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_statement(p):
    '''statement : var_decl
                 | func_decl
                 | if_stmt
                 | loop_stmt
                 | return_stmt
                 | break_stmt
                 | print_stmt
                 | assignment
                 | expression_stmt'''
    p[0] = p[1]


def p_var_decl(p):
    '''var_decl : LET ID ASSIGN expression SEMI'''
    p[0] = VarDecl(p[2], p[4])


def p_func_decl(p):
    '''func_decl : DO ID LPAREN params RPAREN LBRACE statements RBRACE'''
    p[0] = FunctionDecl(p[2], p[4], p[7])

def p_params(p):
    '''params : ID
              | params COMMA ID
              | empty'''
    if len(p) == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_if_stmt(p):
    '''if_stmt : IF expression LBRACE statements RBRACE
               | IF expression LBRACE statements RBRACE ELSE LBRACE statements RBRACE'''
    if len(p) == 6:
        p[0] = IfStatement(p[2], p[4])
    else:
        p[0] = IfStatement(p[2], p[4], p[8])


def p_loop_stmt(p):
    '''loop_stmt : REPEAT LPAREN expression RPAREN LBRACE statements RBRACE'''
    p[0] = RepeatLoop(p[3], p[6])


def p_print_stmt(p):
    '''print_stmt : PRINT LPAREN expression RPAREN SEMI'''
    p[0] = Print(p[3])


def p_assignment(p):
    '''assignment : ID ASSIGN expression SEMI'''
    p[0] = Assignment(p[1], p[3])


def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression MOD expression
                  | expression EQ expression
                  | expression NE expression
                  | expression LT expression
                  | expression GT expression
                  | expression LE expression
                  | expression GE expression'''
    p[0] = BinaryOp(p[2], p[1], p[3])

def p_expression_uminus(p):
    'expression : MINUS expression %prec UMINUS'
    p[0] = -p[2]

def p_expression_group(p):
    '''expression : LPAREN expression RPAREN'''
    p[0] = p[2]

def p_expression_number(p):
    '''expression : NUMBER'''
    p[0] = Number(p[1])

def p_expression_id(p):
    '''expression : ID'''
    p[0] = Identifier(p[1])

def p_expression_func_call(p):
    '''expression : ID LPAREN args RPAREN'''
    p[0] = FunctionCall(p[1], p[3])

def p_args(p):
    '''args : expression
            | args COMMA expression
            | empty'''
    if len(p) == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_expression_stmt(p):
    '''expression_stmt : expression SEMI'''
    p[0] = p[1]


def p_empty(p):
    'empty :'
    p[0] = None

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}'")
    else:
        print("Syntax error at EOF")

def p_return_stmt(p):
    '''return_stmt : RETURN expression SEMI'''
    p[0] = Return(p[2])

def p_break_stmt(p):
    '''break_stmt : BREAK SEMI'''
    p[0] = Break()

def p_expression_input(p):
    '''expression : INPUT LPAREN RPAREN'''
    p[0] = FunctionCall('input', [])

def p_expression_string(p):
    '''expression : STRING'''
    p[0] = String(p[1])

def p_list_literal(p):
    '''expression : LBRACKET elements RBRACKET'''
    p[0] = ListLiteral(p[2])

def p_elements(p):
    '''elements : elements COMMA expression
                | expression
                | empty'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_index_expr(p):
    '''expression : expression LBRACKET expression RBRACKET'''
    p[0] = IndexAccess(p[1], p[3])

def p_expression_typeof(p):
    'expression : TYPEOF LPAREN expression RPAREN'
    p[0] = Typeof(p[3])




#Build the parser using yacc 
#Yet Another Compiler Compiler
parser = yacc.yacc()

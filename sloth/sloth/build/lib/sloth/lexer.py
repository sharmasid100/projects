import ply.lex as lex

# List of token names
tokens = [
    'ID', 'NUMBER', 'STRING',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
    'EQ', 'NE', 'LT', 'GT', 'LE', 'GE', 'ASSIGN',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'COMMA', 'SEMI',
    'LBRACKET' , 'RBRACKET',
]

# Reserved keywords
reserved = {
    'let': 'LET',
    'do': 'DO',
    'if': 'IF',
    'else': 'ELSE',
    'repeat': 'REPEAT',
    'break': 'BREAK',
    'return': 'RETURN',
    'print': 'PRINT',
    'input': 'INPUT',
    'typeof': 'TYPEOF'
}

tokens += list(reserved.values())

# Regular expression rules for simple tokens
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_MOD     = r'%'
t_EQ      = r'=='
t_NE      = r'!='
t_LT      = r'<'
t_GT      = r'>'
t_LE      = r'<='
t_GE      = r'>='
t_ASSIGN  = r'='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACE  = r'\{'
t_RBRACE  = r'\}'
t_COMMA   = r','
t_SEMI    = r';'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'


# Comments: << comment >>
def t_COMMENT(t):
    r'<<[^>]*>>'
    pass  # Ignore comments

# Numbers
def t_NUMBER(t):
    r'\d+(\.\d+)?'
    t.value = float(t.value) if '.' in t.value else int(t.value)
    return t


# Identifiers and keywords
def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t

def t_STRING(t):
    r'\"([^\\\n]|(\\.))*?\"'
    content = t.value[1:-1]  # remove quotes

    # Apply decoding ONLY if it contains a backslash
    if '\\' in content:
        try:
            content = bytes(content, "utf-8").decode("unicode_escape")
        except:
            pass  # if decode fails, keep it raw
    t.value = content
    return t
# Ignored characters (spaces and tabs)
t_ignore = ' \t'

# Newline handling
def t_newline(t):
    '\\n+'
    t.lexer.lineno += len(t.value)

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

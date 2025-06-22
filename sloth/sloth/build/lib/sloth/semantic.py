from sloth.parser import FunctionDecl
from sloth.parser import ListLiteral

class SemanticError(Exception):
    pass

class BreakSignal(Exception):
    pass


class SemanticAnalyzer:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.current_function = None
        self.scopes = [{}]  #stack of variable scopes

    #Scope control
    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()

    #Variable management
    def declare_variable(self, name, vtype):
        if name in self.scopes[-1]:
            raise SemanticError(f"Variable '{name}' already declared in this scope")
        self.scopes[-1][name] = vtype

    def is_declared(self, name):
        return any(name in scope for scope in reversed(self.scopes))

    def visit_Typeof(self, node):
        # Always returns string, as typeof outputs a type name
        self.analyze(node.value)  # Ensure the value is valid
        return "string"


    def assign_variable(self, name, vtype):
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = vtype
                return
        raise SemanticError(f"Assignment to undeclared variable '{name}'")

    def lookup_variable(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if not self.is_declared(name):
            raise SemanticError(f"Variable '{name}' not defined")

    def analyze(self, node):
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise SemanticError(f"No semantic rule for {node.__class__.__name__}")

    def visit_Program(self, node):
        for stmt in node.statements:
            if isinstance(stmt, FunctionDecl):
                if stmt.name in self.functions:
                    raise SemanticError(f"Function '{stmt.name}' already declared")
                self.functions[stmt.name] = stmt
        for stmt in node.statements:
            self.analyze(stmt)

    def visit_VarDecl(self, node):
        if isinstance(node.value, ListLiteral):
            # For list literals, get the specific type
            list_type = self.analyze(node.value)
            self.declare_variable(node.name, list_type)
        else:
            # For non-list variables, infer the type from the value
            value_type = self.analyze(node.value)
            self.declare_variable(node.name, value_type)

    def visit_Assignment(self, node):
        vtype = self.analyze(node.value)
        self.assign_variable(node.name, vtype)

    def visit_Identifier(self, node):
        return self.lookup_variable(node.name)

    def visit_RepeatLoop(self, node):
        cond_type = self.analyze(node.condition)
        if cond_type != "bool":
            raise SemanticError("Loop condition must be boolean")
        self.push_scope()
        for stmt in node.body:
            self.analyze(stmt)
        self.pop_scope()

    def visit_IfStatement(self, node):
        cond_type = self.analyze(node.condition)
        if cond_type != "bool":
            raise SemanticError("If condition must be boolean")
        self.push_scope()
        for stmt in node.then_block:
            self.analyze(stmt)
        self.pop_scope()
        if node.else_block:
            self.push_scope()
            for stmt in node.else_block:
                self.analyze(stmt)
            self.pop_scope()

    def visit_BinaryOp(self, node):
        left_type = self.analyze(node.left)
        right_type = self.analyze(node.right)
        if node.op == "+" and (left_type == right_type or (left_type == "any" or right_type == "any")) :
            return left_type if left_type != "any" else right_type
        
        if node.op in ["<", ">", "==", "!=", "<=", ">="]:
            return "bool"
        raise SemanticError(f"Incompatible {node.op} between {left_type} and {right_type}")

    def visit_ListLiteral(self, node):
        if not node.elements:
            return "list[unknown]"  # or "list[any]" if you prefer
        
        element_types = [self.analyze(e) for e in node.elements]
        base_type = element_types[0]
        
        # Check if all elements have the same type
        if all(t == base_type for t in element_types):
            return f"list[{base_type}]"
        else:
            # Try to find a common supertype if possible
            if all(t in ("int", "float") for t in element_types):
                return "list[float]"
            return "list[any]"

    def visit_IndexAccess(self, node):
        list_type = self.analyze(node.list_expr) 
        index_type = self.analyze(node.index_expr)
        
        #Check if the accessed object is a list/array
        if not (isinstance(list_type, str)):
            raise SemanticError("Indexing a non-list/array type")
        
        # Check if the index is an integer
        if index_type != "int":
            raise SemanticError("List/array index must be an integer")
        
        # Return the element type
        if list_type.startswith("list["):
            return list_type[5:-1]  # Extract the element type from list[type]
        return "any"  # For untyped arrays

    def visit_VarDecl(self, node):
        # Handle array declarations like "let arr = [1, 2, 3]"
        if isinstance(node.value, list):  # Assuming the AST represents arrays as lists
            self.declare_variable(node.name, "array")
        else:
            # For non-array variables, you might want to infer the type from the value
            value_type = self.analyze(node.value)
            self.declare_variable(node.name, value_type)

    def visit_FunctionDecl(self, node):
        self.declare_variable(node.name, "function")
        self.functions[node.name] = node
        self.push_scope()

        # Process parameters with their types
        for param in node.params:
            param_name = param.name if hasattr(param, 'name') else param
            param_type = getattr(param, 'type', "any")

            
            # Special handling for array parameters
            if isinstance(param_type, str) and param_type.startswith("list["):
                # For list parameters, store both the full type and mark as array
                self.declare_variable(param_name, param_type)
            else:
                self.declare_variable(param_name, param_type)

        # Analyze function body
        for stmt in node.body:
            self.analyze(stmt)

        self.pop_scope()

    def visit_FunctionCall(self, node):
        if node.name not in self.functions and not node.name.startswith("input_"):
            raise SemanticError(f"Call to undefined function '{node.name}'")

        # Built-in input functions
        if node.name.startswith("input_"):
            if len(node.args) != 0:
                raise SemanticError("Built-in input functions take no arguments")
            return {"input_string": "string", "input_int": "int", "input_bool": "bool"}.get(node.name, "any")

        func = self.functions[node.name]

        if len(node.args) != len(func.params):
            raise SemanticError(
                f"Function '{node.name}' expects {len(func.params)} arguments, got {len(node.args)}"
            )

        # ✅ Temporarily bind arguments
        self.push_scope()
        for param, arg in zip(func.params, node.args):
            arg_type = self.analyze(arg)
            self.declare_variable(param, arg_type)

        # ✅ Analyze function body WITH bound arguments
        for stmt in func.body:
            self.analyze(stmt)

        self.pop_scope()

        return getattr(func, 'return_type', None)

    def visit_Number(self, node):
        return "int"

    def visit_String(self, node):
        return "string"

    def visit_Boolean(self, node):
        return "bool"
    
    def visit_Print(self, node):
        self.analyze(node.value)
        return None
    
    def visit_Return(self , node):
        value = self.analyze(node.value)
        return value

    def visit_Break(self , node):
        pass
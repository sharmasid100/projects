class InterpreterError(Exception):
    pass

class BreakSignal(Exception):
    pass

class Interpreter:
    def __init__(self):
        self.variables = {}       #Global variable scope
        self.functions = {}       #User-defined functions
        self.return_value = None
        self.in_function = False

    def run(self, node):
        method = 'eval_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_eval)
        return visitor(node)

    def generic_eval(self, node):
        raise InterpreterError(f"No eval rule for {node.__class__.__name__}")

    def eval_Program(self, node):
        for stmt in node.statements:
            result = self.run(stmt)
        return result

    def eval_VarDecl(self, node):
        value = self.run(node.value)
        self.variables[node.name] = value

    def eval_Identifier(self, node):
        if node.name not in self.variables:
            raise InterpreterError(f"Undefined variable '{node.name}'")
        return self.variables[node.name]

    def eval_Assignment(self , node):
        value = self.run(node.value)
        if node.name not in self.variables:
            raise InterpreterError(f"Cannot assign to undeclared variable '{node.name}'")
        self.variables[node.name] = value

    def eval_BinaryOp(self, node):
        left = self.run(node.left)
        right = self.run(node.right)
        op = node.op

        if op == '+': return float(left) + float(right)
        if op == '-': return float(left) - float(right)
        if op == '*': return float(left) * float(right)
        if op == '/': return float(left) / float(right)
        if op == '%': return float(left) % float(right)
        if op == '==': return float(left) == float(right)
        if op == '!=': return float(left) != float(right)
        if op == '<': return float(left) < float(right)
        if op == '<=': return float(left) <= float(right)
        if op == '>': return float(left) > float(right)
        if op == '>=': return float(left) >= float(right)
        if op == '^' : return float(left) ^ int(right)
        raise InterpreterError(f"Unknown binary operator '{op}'")

    def eval_Print(self, node):
        value = self.run(node.value)
        print(value , end = '')


    def eval_FunctionDecl(self, node):
        self.functions[node.name] = node

    def eval_Typeof(self, node):
        value = self.run(node.value)
        if isinstance(value, int):
            return "int"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            if all(isinstance(i, int) for i in value):
                return "list[int]"
            elif all(isinstance(i, str) for i in value):
                return "list[string]"
            else:
                return "list[mixed]"
        else:
            return "unknown"

    def eval_FunctionCall(self, node):
        if node.name == "input_string":
            return input()

        if node.name == "input_int":
            return int(input())

        if node.name == "input_bool":
            val = input().strip().lower()
            return val in ["true", "1", "yes"]

        if node.name not in self.functions:
            raise InterpreterError(f"Undefined function '{node.name}'")

        func = self.functions[node.name]
        if len(node.args) != len(func.params):
            raise InterpreterError("Argument count mismatch")

        old_vars = self.variables.copy()    #Save global scope
        args = [self.run(arg) for arg in node.args]   #Evaluate arguments using global scope
        local_vars = dict(zip(func.params, args))
        #Enter function scope
        self.variables = local_vars
        self.in_function = True

        for stmt in func.body:
            self.run(stmt)
            if self.return_value is not None:
                ret = self.return_value
                self.return_value = None
                self.in_function = False
                self.variables = old_vars
                return ret

        self.variables = old_vars
        self.in_function = False

    def eval_RepeatLoop(self, node):
        while self.run(node.condition):
            try:
                for stmt in node.body:
                    self.run(stmt)
            except BreakSignal:
                break


    def eval_IfStatement(self, node):
        if self.run(node.condition):
            for stmt in node.then_block:
                self.run(stmt)
        elif node.else_block:
            for stmt in node.else_block:
                self.run(stmt)

    def eval_Number(self, node):
        return node.value

    def eval_String(self, node):
        return node.value
    
    def eval_ListLiteral(self, node):
        return [self.run(elem) for elem in node.elements]

    def eval_IndexAccess(self, node):
        lst = self.run(node.list_expr)
        idx = self.run(node.index_expr)
        idx = int(idx)
        return lst[idx]

    def eval_IndexAssign(self, node):
        lst = self.run(node.list_expr)
        idx = self.run(node.index_expr)
        val = self.run(node.value)
        lst[idx] = val

    def eval_Return(self, node):
        self.return_value = self.run(node.value)

    def eval_Break(self, node):
        raise BreakSignal()






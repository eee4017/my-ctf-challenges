import argparse
import collections
import dataclasses
import itertools
import json
import logging
import os
import random
import sys
import tempfile
import typing
import lark
import string
from enum import Enum

random.seed(0)

l = logging.getLogger("compiler")
PARSER = None
GRAMMAR_FILE = 'grammar.lark'
with open(GRAMMAR_FILE, 'r') as grammar:
    PARSER = lark.Lark(grammar, start='program')



OP_TO_ASM = {
    'add': 'mpl::plus',
    'sub': 'mpl::minus',
    'mul': 'mpl::times',
    'div': 'mpl::divides',
    'mod': 'mpl::modulus',
    'bit_lor': 'mpl::or_',
    'bit_land': 'mpl::and_',
    'bit_and': 'mpl::bitand_',
    'bit_xor': 'mpl::bitxor_',
    'bit_or': 'mpl::bitor_',
    'bit_shift_left': 'mpl::shift_left',
    'bit_shift_right': 'mpl::shift_right',
    'equals': 'mpl::equal_to',
    'not_equals': 'mpl::not_equal_to',
    'less_than' : 'mpl::less',
    'greater_than' : 'mpl::greater',
    'less_than_or_equal': 'mpl::less_equal',
    'greater_than_or_equal': 'mpl::greater_equal',
}

OP_RETURN_BOOL = ['bit_lor', 'bit_land', 'equals', 'not_equals', 'less_than', 'greater_than', 'less_than_or_equal', 'greater_than_or_equal']

@dataclasses.dataclass
class Function:
    name: str
    scope: str
    arg_list: typing.List[str]
    is_special: bool = False
    has_return_value: bool = True
    is_external: bool = False

    def arg_name(self, idx):
        name = f"_{self.name}_arg_{idx}"
        if self.is_external:
            name += "_export"
        return name

    def return_location(self):
        name = f"_{self.name}_return_location"
        if self.is_external:
            name += "_export"
        return name

class Type(Enum):
    NONE = 0
    INT = 1
    BOOL = 2
    VECTOR = 3
    MAYBE_INT = 4
    MAYBE_BOOL = 5

class Value():
    type_: Type
    name: str

    def __init__(self, type:Type, name:str):
        self.type_ = type
        self.name = name

    def ismaybe(self):
        return self.type_ == Type.MAYBE_INT or self.type_ == Type.MAYBE_BOOL

    def type(self):
        return self.type_
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


SPECIAL_FUNCTIONS = [
Function("OUT", "", ["output"], True, False),
Function("IN", "", ["input"], True, True),
Function("VECTOR", "", [], True, True),
Function("SET_VECTOR", "", ["map", "at", "value"], True, False),
]

GENERATE_ASSEMBLY_SPECIAL_FUNCTIONS = {'OUT': lambda args: f"typedef {args[0]} type;\n",
                                       'IN': lambda args: f"",
                                       'VECTOR': lambda args: f"",
                                       'SET_VECTOR': lambda args: f"",
}

obfuscate = False

class ExtractFunctionsPass(lark.visitors.Interpreter):
    functions = list()
    scope = list()
    
    def function_def(self, tree):
        function_name = str(tree.children[0].children[0])
        args = [str(arg.children[0]) for arg in tree.children[1].children]
        scope = ":".join(self.scope)

        self.functions.append(Function(function_name, scope, args))

        self.scope.append(function_name)
        self.visit_children(tree)
        self.scope.pop()

    def extern_statement(self, tree):
        function_name = str(tree.children[0].children[0])
        args = [str(arg.children[0]) for arg in tree.children[1].children]
        scope = ":".join(self.scope)

        self.functions.append(Function(function_name, scope, args, is_external=True))

class UsedVariablesPass(lark.Transformer):

    def __init__(self):
        self.count_lhs = True

    def _flatten(self, tree):
        to_return = list(itertools.chain.from_iterable(tree))
        done = False
        while not done:
            if all(not isinstance(element, list) for element in to_return):
                done = True
            else:
                to_return = list(itertools.chain.from_iterable(to_return))
        return to_return


    statements = _flatten
    program = _flatten
    function_def = _flatten
    conditional = _flatten
    while_loop = _flatten
    sync = _flatten
    elif_clauses = _flatten
    function_call = _flatten
    else_clause = _flatten
    expression_function_arg_list = _flatten
    return_statement = _flatten
    array_access = _flatten

    arg_list = lambda self, _: list()
    function_name = lambda self, _: list()
    number = lambda self, _: list()
    hexnumber = lambda self, _: list()
    string  = lambda self, _: list()
    outs_literal = lambda self, _: list()
    extern_statement = lambda self, _: list()
    asm_statement = lambda self, _: list()

    def lhs(self, tree):
        if self.count_lhs:
            return tree[0]
        else:
            return []

    def binary_operation(self, tree):
        tree[2].extend(tree[0])
        return tree[2]

    def assignment(self, tree):
        tree[0].extend(tree[1])
        return tree[0]

    def var_name(self, tree):
        return [str(tree[0])]

def not_implemented(func):
    def wrapper(*args, **kwargs):
        l.warning(f"{func.__name__} is not implemented")
        func(*args, **kwargs)
    return wrapper

cnt = -1
def random_cname_generator(length=5):
    global cnt
    cnt += 1
#     return f"${str(cnt).rjust(length, '0')}"
    ret = [random.choice(string.ascii_letters)]
    ret.extend([random.choice(string.ascii_letters + string.digits) for i in range(length - 1)])
    return ''.join(ret)

#cnt = -1
#def random_cname_generator(length=5):
#    global cnt
#    cnt += 1
#    return f"var__{cnt}"


WHILE_NAME = random_cname_generator(7)
class GenerateAssemblyPass(lark.visitors.Interpreter):
    functions: typing.Dict[str, Function]
    to_return = list()
    variables = [dict()]
    variable_length = 7
    global_num = 0
    return_variables = list()
    function_name_dict = dict()

    def function_name_obfuscate(self,name):
        if name not in self.function_name_dict:
            if obfuscate:
                self.function_name_dict[name] = random_cname_generator(self.variable_length)
            else:
                self.function_name_dict[name] = name                

        return self.function_name_dict[name]


    def emit(self, assembly):
        self.to_return.append(assembly)
        
    def _new_temp_variable(self):
        self.global_num += 1
        return f"tmp_{self.global_num}"
    
    def define_variable(self, key:str, type:Type):
        if obfuscate:
            new_var = random_cname_generator(self.variable_length)
        else:
            if not key:
                self.global_num += 1
                new_var =  f"tmp_{self.global_num}"
            else:
                self.global_num += 1
                new_var =  f"{key}_{self.global_num}"
        self.variables[-1][key] = Value(type=type, name=new_var)
        return self.variables[-1][key]
    
    def __init__(self, functions):
        self.functions = {f.name: f for f in functions}
        self._num = [0]

    def handle_assignment(self, name, r: Value):
        l = self.variables[-1][name]
        self.variables[-1][name] = r

        return
        if l.type() == r.type():
            self.variables[-1][name] = r
        elif l.ismaybe() and (not r.ismaybe()):
            new = self.define_variable(None, Type.MAYBE_INT)
            self.emit(f"typedef just<{r}>::type {new};")
            # self.variables[-1][name] = new
        elif (not l.ismaybe()) and r.ismaybe():
            new = self.define_variable(name, Type.INT)
            self.emit(f"typedef typename {r}::type::type {new};")
            # self.variables[-1][name] = new
        else:
            l.error(f"Don't know how to handle this assignment {l.type()} {r.type()}")


    def code_block(self, tree, used_variables):
        code_block_used_variables = set(UsedVariablesPass().transform(tree))
        code_block_state = self.define_variable("code_block_state", Type.VECTOR)
        the_code_block = self.define_variable(f"code_block", Type.VECTOR)

        self.variables.append(self.variables[-1].copy())
        self._num.append(0)
        self.emit(f"""
        template <typename {code_block_state}> 
        struct {the_code_block} {{
        """)

        # resolve variable
        for i, v in enumerate(used_variables):
            if v in code_block_used_variables:
                var_name = self.define_variable(v, Type.INT)
                self.variables[-1][v] = var_name
                self.emit(f"typedef typename mpl::at<{code_block_state}, mpl::integral_c<int, {i}>>::type {var_name};")
        
        self.visit(tree)
        self.emit(f"typedef typename mpl::vector<{', '.join([ str(self.variables[-1][v]) for v in used_variables ])}>::type type;")
        
        self.emit(f"""
        }};
        """)

        self.variables.pop()
        self._num.pop()
        return the_code_block
    
    def predicate(self, tree, used_variables):
        self.variables.append(self.variables[-1].copy())
        self._num.append(0)

        predicate_used_variables = set(UsedVariablesPass().transform(tree))
        predicate_state = self.define_variable("predicate_state", Type.VECTOR)
        predicate = self.define_variable(f"predicate", Type.VECTOR)
        self.emit(f"""
        template <typename {predicate_state}> 
        struct {predicate} {{
        """)
        # save the state
        # original = {v:self.variables[-1][v] for v in predicate_used_variables}
        for i, v in enumerate(used_variables):
            if v in predicate_used_variables:
                var_name = self.define_variable(v, Type.INT)
                # self.variables[-1][v] = var_name
                self.emit(f"typedef typename mpl::at<{predicate_state}, mpl::integral_c<int, {i}>>::type {var_name};")
        
        ret = self.visit(tree)
        if ret.type() == Type.MAYBE_BOOL:
            self.emit(f"typedef typename {ret}::type type;")
        else:
            l.error(f"type error: predicate must return bool, but it return {ret.type()}")
        
        self.emit(f"""
        }};
        """)
        # for k, v in original.items():
        #     self.variables[k] = v    

        self.variables.pop()
        self._num.pop()
        return predicate
    
    def while_loop(self, tree):
        # prior_variables = self.variables.copy()
        # undefined_before_loop = set()
        
        used_variables = set(UsedVariablesPass().transform(tree.children[0]))
        used_variables.update(UsedVariablesPass().transform(tree.children[1]))

        # we don't have to handle any constant variables
        # used_variables = list(filter(lambda var: not var in self.constant_variables, used_variables))
        
        code = self.code_block(tree.children[1], used_variables)
        pred = self.predicate(tree.children[0], used_variables)
        
        state = self.define_variable("while_state", Type.VECTOR)
        state_vector = [ str(self.variables[-1][v]) for v in used_variables ]
        self.emit(f"typedef mpl::vector<{', '.join(state_vector)}> {state};")
        
        while_ = self.define_variable(f"while", Type.VECTOR)
        self.emit(f"""
        typedef typename {WHILE_NAME}<
          {pred}<mpl::_1>, 
          mpl::quote1<{code}>,
          {state}
        >::type {while_};
        """)
        
        # uncompress variable
        for i, v in enumerate(used_variables):
            # var_name = self.define_variable(v, Type.MAYBE_INT)
            # self.emit(f"typedef typename main_::bind2<mpl::quote2<mpl::at>, main_::just<{if_}>, main_::just<mpl::integral_c<int, {i}>>>::type {var_name};")
            var_name = self.define_variable(v, Type.INT)
            self.emit(f"typedef typename mpl::at<{while_}, mpl::integral_c<int, {i}>>::type {var_name};")

    def bind_code_block_with_state(self, code_block, state):
        binder = self.define_variable(f"binder", Type.VECTOR)
        self.emit(f"""
        struct {binder} {{
            typedef typename {code_block}<{state}>::type type;
        }}; 
        """)
        return binder
    
    def conditional(self, tree):
        used_variables = set(UsedVariablesPass().transform(tree.children[0]))
        used_variables.update(UsedVariablesPass().transform(tree.children[1]))
        # used_variables = list(filter(lambda var: not var in self.constant_variables, used_variables))
        
        state = self.define_variable("if_state", Type.VECTOR)
        state_vector = [ str(self.variables[-1][v]) for v in used_variables ]
        self.emit(f"typedef mpl::vector<{', '.join(state_vector)}> {state};")

        pred = self.predicate(tree.children[0], used_variables)
        code_true = self.code_block(tree.children[1], used_variables)
        code_true_binder = self.bind_code_block_with_state(code_true, state)
        has_else = (len(tree.children) == 3)
        
        if has_else:
            code_false = self.code_block(tree.children[2], used_variables)
            code_false_binder = self.bind_code_block_with_state(code_false, state)

        
        if_ = self.define_variable("if_output", Type.VECTOR)
        self.emit(f"""
        typedef typename mpl::eval_if<
          typename {pred}<{state}>::type, 
          {code_true_binder},
          { f"{code_false_binder}" if has_else else state }
        >::type {if_};
        """)
        
        # uncompress variable
        for i, v in enumerate(used_variables):
            # var_name = self.define_variable(v, Type.MAYBE_INT)
            # self.emit(f"typedef typename main_::bind2<mpl::quote2<mpl::at>, main_::just<{if_}>, main_::just<mpl::integral_c<int, {i}>>>::type {var_name};")
            var_name = self.define_variable(v, Type.INT)
            self.emit(f"typedef typename mpl::at<{if_}, mpl::integral_c<int, {i}> >::type {var_name};")
        
    def return_statement(self, tree):
        return_expression = self.visit(tree.children[0])
        self.return_variables.append(return_expression)

    def function_def(self, tree):
        function_name = str(tree.children[0].children[0])
        args = [str(arg.children[0]) for arg in tree.children[1].children]
        function = self.functions[function_name]
        self.variables.append(self.variables[-1].copy())

        args = [ f"typename {self.define_variable(arg, Type.INT)}" for arg in args ]
        self.emit(f"template <{','.join(args)}>")
        self.emit(f"struct {self.function_name_obfuscate(function_name)} {{")
        
        self.visit_children(tree.children[2])

        if len(self.return_variables) == 0:
            l.error(f"Function {function_name} does not have any return statements.")
            sys.exit(-1)
        
        the_return = self.return_variables.pop()
        self.emit(f"typedef {the_return} type;")
        self.emit(f"}};")
        self.variables.pop()
        

    
    def var_name(self, tree):
        var_name = str(tree.children[0])
        print(self.variables[-1])
        # if var_name in self.constant_variables:
        #     return self.visit(self.constant_variables[var_name])
        if var_name in self.variables[-1]:
            return self.variables[-1][var_name]
        else:
            l.error(f"variable {var_name} not defined.")

    @staticmethod
    def array_access_value_to_ref(value):
        value_prefix = "_value"
        original = value[:-len(value_prefix)]
        return f"{original}_ref"
    
    def array_access(self, tree):
        array = self.visit(tree.children[0])
        expression_result = self.visit(tree.children[1])
        new_variable_name = self.define_variable(f"{array}_access_{self._new_temp_variable()}", Type.INT)
        new_variable_name_value = self.define_variable(f"{new_variable_name}_value", Type.INT)
        print("array_access", array, expression_result)
        self.emit(f"typedef typename mpl::at<{array}, {expression_result}>::type {new_variable_name_value};")
        return new_variable_name_value

    @not_implemented
    def outs_literal(self, tree):
        pass

    def function_call(self, tree):
        function_name = str(tree.children[0].children[0])

        if function_name == "define_constant":
            if len(tree.children[1].children) != 2:
                l.error("Call to {function_name} has the wrong number of arguments {len(tree.children[1].children)} expected 2")
                sys.exit(-1)
            if tree.children[1].children[0].data != 'var_name':
                l.error("define_constant first argument must be var_name, instead had {tree.children[1].children[0]}")
                sys.exit(-1)

            var_name = str(tree.children[1].children[0].children[0])
            const_expression = tree.children[1].children[1]

            if var_name in self.variables:
                l.error("define_constant trying to define {var_name} but it is already defined as a variable.")
                sys.exit(-1)

            # if var_name in self.constant_variables:
            #     l.error("define_constant trying to define {var_name} but it is already defined as a constant variable.")
            #     sys.exit(-1)

            # self.constant_variables[var_name] = const_expression
            return None

        function_args = [self.visit(c) for c in tree.children[1].children] if len(tree.children) == 2 else []
        
        if function_name in self.functions:
            function = self.functions[function_name]
            if len(function_args) != len(function.arg_list):
                l.error(f"Call to function {function_name} has the wrong number of arguments {len(function_args)}, expected {len(function.arg_list)}")
                sys.exit(-1)

            if function.is_special:
                l.debug(f"Special function call {function_name}")
                if function.has_return_value:
                    # to_send = list(function_args)
                    # to_send.append(return_var)
                    if function_name == "VECTOR":
                        return_var = self.define_variable(f"vector_{self._new_temp_variable()}", Type.VECTOR)
                        self.emit(f"typedef mpl::map<>::type {return_var};")
                        return return_var
                    elif function_name == "IN":
                        v_name = tree.children[1].children[0].children[0]
                        return_var = self.define_variable(f"in_{self._new_temp_variable()}", Type.INT)
                        self.emit(f"typedef mpl::integral_c<int, __IN_{v_name}> {return_var};")
                        return return_var
                else:
                    if function_name == "OUT":
                        var = function_args[0]
                        print("var", var)
                        if var.type() == Type.INT: 
                            self.emit(f"typedef {var}::type type;")
                        elif var.type() == Type.MAYBE_INT:
                            self.emit(f"typedef {var}::type::type type;")
                        else:
                            l.error(f"type error {var.type()}")  
                    elif function_name == "SET_VECTOR":
                        v_name = tree.children[1].children[0].children[0]
                        ma = function_args[0]
                        at = function_args[1]
                        val = function_args[2]
                        print("*****", ma, at.type(), val.type())
                        temp_map = self.define_variable(f"vector_{self._new_temp_variable()}", Type.VECTOR)
                        self.emit(f"typedef typename mpl::insert<{ma}, mpl::pair<{at}, {val}>>::type {temp_map};")
                        self.variables[-1][v_name] = temp_map
                    else:
                        l.error(f"Don't know how to handle this function {function_name}")
                    return None
            else:
                l.debug(f"Normal fuction call {function_name}")
                return_var = self.define_variable(self._new_temp_variable(), Type.INT)
                
                self.emit(f"typedef typename {self.function_name_obfuscate(function_name)}<{','.join([str(a) for a in function_args])}>::type {return_var};")
                return return_var
        else:
            l.error(f"Call to undefined function {function_name}.")
            sys.exit(-1)
    
    def assignment(self, tree):
        expression_result = self.visit(tree.children[1])
        
        if tree.children[0].children[0].data == "var_name":
            variable_name = str(tree.children[0].children[0].children[0])
            # if variable_name in self.constant_variables:
            #     l.error("Cannot assign to constant variable {variable_name}")
            #     sys.exit(-1)
            

            if variable_name not in self.variables[-1]:
                self.variables[-1][variable_name] = expression_result
                # new_variable_name = self.define_variable(key=variable_name, type=Type.INT)
                # print(str(expression_result))
                # print(variable_name, str(new_variable_name))
                # self.emit(f"typedef {expression_result}::type {new_variable_name};")
            else:
                self.handle_assignment(variable_name, expression_result)
                # new_variable_name = self.define_variable(variable_name, Type.INT)
                # self.emit(f"typedef {expression_result} {new_variable_name};")
                
                
            return None
        # elif tree.children[0].children[0].data == "array_access":
        #     array_value = self.visit(tree.children[0].children[0])
        #     array_ref = GenerateAssemblyPass.array_access_value_to_ref(array_value)
        #     self.emit(f"_ = AST {array_ref} {expression_result}")
        #     return None
        else:
            l.error(f"Don't know how to handle this assignment statement {tree}")
            sys.exit(-1)

    def binary_operation(self, tree):
        op = tree.children[1].data

        assembly_op = OP_TO_ASM[op]

        if op in OP_RETURN_BOOL:
            return_type = Type.MAYBE_BOOL
        else:
            return_type = Type.MAYBE_INT
                
        left_var = self.visit(tree.children[0])
        right_var = self.visit(tree.children[2])
        new_variable = self.define_variable(None, return_type)

        print(new_variable, assembly_op, left_var, right_var)
        def maybefy(var):
            if var.type() == Type.INT:
                var = f"main_::just<{var}>"
            elif var.type() == Type.MAYBE_INT or var.type() == Type.MAYBE_BOOL:
                var = f"{var}"
            else:
                l.error(f"maybefy type error: {var} {var.type()}")
            return var

        # self.emit(f"typedef typename main_::bind2<mpl::quote2<{assembly_op}>, {maybefy(left_var)}, {maybefy(right_var)}>::type {new_variable};")
        self.emit(f"typedef typename {assembly_op}<{left_var}, {right_var}>::type {new_variable};")
        return new_variable

    def number(self, tree):
        # For now we do this very inefficient thing where we DUP each
        # literal string (so that the binary operator doesn't have to
        # worry about which literal can go where)
        
        num = tree.children[0]
        return Value(type=Type.INT, name=f"mpl::integral_c<int, {num}>")

    @not_implemented
    def _create_num_literal(self, num, new_variable):
        pass

    @not_implemented
    def hexnumber(self, tree):
        num = int(tree.children[0], 16)
        new_variable = self._new_temp_variable()
        self._create_num_literal(num, new_variable)
        return new_variable
    
    @not_implemented
    def string(self, tree):
        the_str = eval(str(tree.children[0]))
        if len(the_str) > 8:
            l.error(f"string literal {the_str} is larger than the limit of 8.")
            sys.exit(-1)

        the_str_binary = the_str.encode()

        int_repr = int.from_bytes(the_str_binary, 'little')
        new_variable = self._new_temp_variable()
        self._create_num_literal(int_repr, new_variable)
        return new_variable
    
    @not_implemented
    def export_statement(self, tree):
        pass
    
    @not_implemented
    def extern_statement(self, tree):
        pass
    
    @not_implemented
    def asm_statement(self, tree):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="compiler")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("--input", type=str, required=True, help="The file to compile")
    parser.add_argument("--output", type=str, help="Where to write the output.")
    parser.add_argument("--obfuscate", action='store_true', help="Enable obfuscater")

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.obfuscate:
        obfuscate = True

    with open(args.input, 'r') as f:
        original_file_input = f.read()
        tree = PARSER.parse(original_file_input)
        
    extract_functions = ExtractFunctionsPass()
    extract_functions.visit(tree)
    defined_functions = extract_functions.functions
    # print(f"found functions={extract_functions.functions}")

    defined_functions.extend(SPECIAL_FUNCTIONS)
    print(tree.pretty())

    generate_assembly = GenerateAssemblyPass(defined_functions)
    generate_assembly.visit(tree)

    Pred = random_cname_generator(7)
    Func = random_cname_generator(7)
    Value = random_cname_generator(7)
    go = random_cname_generator(7)
    lazy_apply = random_cname_generator(7)
    F = random_cname_generator(7)
    X = random_cname_generator(7)

    assembly_code = f"""
    #include <iostream>
    #include <mpl_all.hpp>

    struct _ {{
        template <typename {Pred}, 
                typename {Func},
                typename {Value}> 
        struct {WHILE_NAME}{{
        template <typename {F}, typename {X}> struct {lazy_apply};

        template <typename {X}>
        struct {go} : mpl::eval_if<typename mpl::apply<{Pred}, {X}>::type, {lazy_apply}<{Func}, {X}>,
                            mpl::identity<{X}>> {{}};

        template <typename {F}, typename {X}> struct {lazy_apply} {{
            typedef typename {go}<typename mpl::apply<{F}, {X}>::type>::type type;
        }};

        typedef typename {go}<{Value}>::type type;
        }};

    """.splitlines()
    assembly_code.extend(generate_assembly.to_return)
    assembly_code.extend( (f"""
    }};

    int main() {{
    std::cout << _::type() << '\\n';
    }}
    """).splitlines())


    with open(args.output, 'w') as f:
        f.write('\n'.join(assembly_code))
        
    os.system(f"clang-format -i {args.output}")
    # print(generate_assembly.variables)

"""6.009 Lab 9: Carlae Interpreter Part 2"""

import sys

sys.setrecursionlimit(10_000)

###########################
# Carlae-related Exceptions #
###########################


class CarlaeError(Exception):
    """
    A type of exception to be raised if there is an error with a Carlae
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class CarlaeSyntaxError(CarlaeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class CarlaeNameError(CarlaeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class CarlaeEvaluationError(CarlaeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    CarlaeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(x):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


parenthesis = ["(", ")"]
keywords = [":=", "function"]
whitespace = [" ", "\n"]
comment = "#"
delimiters = [*parenthesis, *whitespace]


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Carlae
                      expression
    """
    tokens = []

    current_token = ""
    is_comment = False

    # loop thr the chars in source
    for char in source:
        if char in delimiters:
            # put current token into list of tokens if it exists
            if current_token:
                tokens.append(current_token)

            current_token = ""

        # ignore whitespaces
        if char in whitespace:
            if char == "\n":
                is_comment = False
            continue

        if is_comment:
            continue

        # add parenthesis then move on
        if char in parenthesis:
            tokens.append(char)

            continue

        # assuming no string type and # is not in string
        if char == comment:
            is_comment = True
            continue

        # build up current token
        current_token += char

        # check if current token is a keyword
        if current_token in keywords:
            tokens.append(current_token)
            current_token = ""

    # if remaining token, add to tokens
    if current_token:
        tokens.append(current_token)

    return tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    # base case -> one token
    if len(tokens) == 1:
        # edge case -> single parenthesis
        if tokens[0] == "(" or tokens[0] == ")":
            raise CarlaeSyntaxError()

        return number_or_symbol(tokens[0])

    # recursive case
    groups = _group_tokens(tokens)
    parsed_expression = [parse(group) for group in groups]

    return parsed_expression


def _group_tokens(tokens):
    """ """
    # check if enclosed in parenthesis
    if tokens[0] != "(" or tokens[-1] != ")":
        raise CarlaeSyntaxError()

    inner_tokens = tokens[1:-1]
    groups = []
    opening_parenthesis_index = -1
    paren_stack = 0

    for i, token in enumerate(inner_tokens):
        if token == "(":
            paren_stack += 1
            if opening_parenthesis_index < 0:
                opening_parenthesis_index = i

            continue

        if token == ")":

            paren_stack -= 1

            if paren_stack < 0:
                raise CarlaeSyntaxError()

            if paren_stack == 0:
                groups.append(inner_tokens[opening_parenthesis_index : i + 1])
                opening_parenthesis_index = -1

            continue

        if opening_parenthesis_index >= 0:
            continue

        groups.append([token])

    if paren_stack > 0:
        raise CarlaeSyntaxError()

    return groups


######################
# Built-in Functions #
######################


def multiply(args):
    value = 1

    for arg in args:
        value *= arg

    return value


def divide(args):
    value = args[0]

    for arg in args[1:]:
        value /= arg

    return value


def assignment(variable, value, env):
    env.assign(variable, value)

    return value


def _all_satifies(args, flag):
    if args == []:
        return True

    first = args[0]

    for arg in args[1:]:
        if not flag(first, arg):
            return False

        first = arg

    return True


def is_equal(args):
    return _all_satifies(args, lambda x, y: x == y)


def is_greater(args):
    return _all_satifies(args, lambda x, y: x > y)


def is_greater_or_equal(args):
    return _all_satifies(args, lambda x, y: x >= y)


def is_less(args):
    return _all_satifies(args, lambda x, y: x < y)


def is_less_or_equal(args):
    return _all_satifies(args, lambda x, y: x <= y)


def eval_not(args):
    # print("args")
    # print(args)

    if len(args) != 1:
        # print("raising error")
        raise CarlaeEvaluationError()

    return not args[0]


def create_pair(args):
    if len(args) != 2:
        raise CarlaeEvaluationError()

    head, tail = args

    return Pair(head, tail)


class Pair:
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail

    def __str__(self):
        return f"(head: {self.head}, tail: {self.tail})"


def get_head(args):
    if len(args) != 1:
        raise CarlaeEvaluationError()

    pair = args[0]

    if not isinstance(pair, Pair):
        raise CarlaeEvaluationError()

    return pair.head


def get_tail(args):
    if len(args) != 1:
        raise CarlaeEvaluationError()

    pair = args[0]

    if not isinstance(pair, Pair):
        raise CarlaeEvaluationError()

    return pair.tail


class Nil:
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Nil)


def create_list(args):
    if len(args) == 0:
        return Nil()

    return Pair(args[0], create_list(args[1:]))


def _is_list(obj):
    if obj == Nil():
        return True

    if type(obj) is not Pair:
        return False

    # if obj.head:
    return _is_list(obj.tail)

    # return False


def render_list(lst):
    render = "("

    current = lst

    while current != Nil():

        render += str(current.head)

        if current.tail != Nil():
            render += " "

        current = current.tail

    return render + ")"


def is_list(args):
    arg = args[0]

    return _is_list(arg)


def get_length(args):
    lst = args[0]

    if not _is_list(lst):
        raise CarlaeEvaluationError()

    if lst == Nil():
        return 0

    return 1 + get_length([lst.tail])


def _get_index(list_or_pair, index):
    if type(index) not in [int, float]:
        raise CarlaeEvaluationError()

    if _is_list(list_or_pair):
        if list_or_pair == Nil() and index != 0:
            raise CarlaeEvaluationError()

        if index == 0:
            # empty list
            if list_or_pair == Nil():
                raise CarlaeEvaluationError()

            return list_or_pair.head

        return _get_index(list_or_pair.tail, index - 1)

    if type(list_or_pair) == Pair:
        if index == 0:
            return list_or_pair.head

    raise CarlaeEvaluationError()


def get_index(args):
    if len(args) != 2:
        return CarlaeEvaluationError()

    list_or_pair, index = args

    return _get_index(list_or_pair, index)


def _is_all_lists(lsts):
    for lst in lsts:
        if not _is_list(lst):
            return False

    return True


def copy_list(lst):
    if lst == Nil():
        return Nil()

    return Pair(lst.head, copy_list(lst.tail))


def get_last_pair(pair):
    if pair.tail == Nil():
        return pair

    return get_last_pair(pair.tail)


def _concat(lsts):
    if len(lsts) == 0:
        return Nil()

    copied_head = copy_list(lsts[0])
    concat_body = _concat(lsts[1:])

    if copied_head == Nil():
        return concat_body

    get_last_pair(copied_head).tail = concat_body

    return copied_head


def concat(args):
    # check if all args are lists
    is_all_lists = _is_all_lists(args)

    if not is_all_lists:
        raise CarlaeEvaluationError()

    return _concat(args)


def _call_func(func, args):
    if callable(func):
        return func(args)
    else:
        return func.call(args)


def _map(func, lst):
    if lst == Nil():
        return Nil()

    current_val = lst.head
    mapped_val = _call_func(func, [current_val])

    return Pair(mapped_val, _map(func, lst.tail))


def check_is_func_and_list(args):
    if len(args) != 2:

        raise CarlaeEvaluationError()

    func, lst = args

    is_valid_args = (callable(func) or isinstance(func, CarlaeFunction)) and _is_list(
        lst
    )

    if not is_valid_args:
        raise CarlaeEvaluationError()

    return func, lst


def map_list(args):
    func, lst = check_is_func_and_list(args)

    return _map(func, lst)


def _filter(func, lst):
    if lst == Nil():
        return Nil()

    current_value = lst.head
    should_include = _call_func(func, [current_value])

    if should_include:
        return Pair(current_value, _filter(func, lst.tail))

    return _filter(func, lst.tail)


def filter_list(args):
    func, lst = check_is_func_and_list(args)

    return _filter(func, lst)


def _reduce(func, lst, initial):
    if lst == Nil():
        return initial

    new_initial = _call_func(func, [initial, lst.head])

    return _reduce(func, lst.tail, new_initial)


def reduce_list(args):
    if len(args) != 3:
        raise CarlaeEvaluationError()

    func, lst = check_is_func_and_list(args[:2])
    initial = args[2]

    return _reduce(func, lst, initial)


def begin(args):
    return args[-1]


carlae_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": divide,
    "@t": True,
    "@f": False,
    "=?": is_equal,
    ">": is_greater,
    ">=": is_greater_or_equal,
    "<": is_less,
    "<=": is_less_or_equal,
    "not": eval_not,
    "pair": create_pair,
    "head": get_head,
    "tail": get_tail,
    "nil": Nil(),
    "list": create_list,
    "list?": is_list,
    "length": get_length,
    "nth": get_index,
    "concat": concat,
    "map": map_list,
    "filter": filter_list,
    "reduce": reduce_list,
    "begin": begin,
}


##############
# Evaluation #
##############


class Environment:
    def __init__(self, init_bindings, parent_env=None):
        self.variables = init_bindings
        self.parent = parent_env

    def get(self, variable):
        # get the variable binding
        value = self.variables.get(variable)

        # if binding exists
        if value != None:
            return value

        # binding doesn't exist, check if parent env exists
        if not self.parent:
            raise CarlaeNameError()
            # return None

        # get variable from parent env
        return self.parent.get(variable)

    def assign(self, variable, value):
        """
        Set a variable binding in the current env
        """

        self.variables[variable] = value

    def set(self, variable, value):
        if variable in self.variables:
            self.variables[variable] = value
            return

        if not self.parent:
            raise CarlaeNameError()

        return self.parent.set(variable, value)

    def delete(self, variable):
        if variable in self.variables:
            value = self.variables[variable]

            del self.variables[variable]

            return value

        raise CarlaeNameError()

    def __str__(self):
        return f"(variables: {self.variables})"


def _create_env(env):
    if env:
        return env

    builtins = Environment(carlae_builtins)

    return Environment({}, builtins)


class CarlaeFunction:
    def __init__(self, parameters, body, enclosing_env):
        self.parameters = parameters
        self.body = body
        self.enclosing_env = enclosing_env

    def call(self, arguments):

        # different number of parameters -> error
        if len(self.parameters) != len(arguments):
            # print(self.parameters)
            # print(arguments)
            raise CarlaeEvaluationError()

        # evaluate arguments of function
        evaluated_arguments = [evaluate(arg, self.enclosing_env) for arg in arguments]

        # create environment for function
        func_env = Environment({}, self.enclosing_env)

        # bind variables to environment
        for var, val in zip(self.parameters, evaluated_arguments):
            func_env.assign(var, val)

        # evalute the function body in the function environment

        return_value = evaluate(self.body, func_env)

        return return_value

    def __str__(self):
        return "function_object"
        # return f"parameters: {self.parameters}, body: {self.body}"


def create_function(parameters, body, enclosing_env):
    return CarlaeFunction(parameters, body, enclosing_env)


# condtionals
class ConditionalBinOp:
    def __init__(self, statements, env) -> None:
        self.statements = statements
        self.env = env


class And(ConditionalBinOp):
    def eval(self):
        for statement in self.statements:
            value = evaluate(statement, self.env)
            # print(value)

            if not value:
                return False

        return True


class Or(ConditionalBinOp):
    def eval(self):
        for statement in self.statements:
            value = evaluate(statement, self.env)

            if value:
                return True

        return False


def evaluate(tree, env=None):
    """
    Evaluate the given syntax tree according to the rules of the Carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """

    env = _create_env(env)

    if tree == []:
        raise CarlaeEvaluationError()

    if isinstance(tree, str):
        return env.get(tree)

    if not isinstance(tree, list):
        return tree

    keyword = tree[0]

    # variable assignment
    if keyword == ":=":
        # handle function assignment -> shorthand
        is_shorthand_func_def = isinstance(tree[1], list)

        # get parts from assignment expression
        _, variable, expression = tree

        if is_shorthand_func_def:
            # print("is short hand function def")

            func_name = tree[1][0]
            parameters = tree[1][1:]
            body = tree[2]

            func = create_function(parameters, body, env)

            return assignment(func_name, func, env)

        evaluated_expression = evaluate(expression, env)

        # set variable binding
        return assignment(variable, evaluated_expression, env)
    elif keyword == "function":
        # get parameters and body of function
        _, parameters, body = tree

        return create_function(parameters, body, env)
    elif keyword == "if":
        _, condition, statement1, statement2 = tree

        # evaluate the condtion
        condition_value = evaluate(condition, env)

        truthy_val = condition_value

        if truthy_val:
            return evaluate(statement1, env)

        return evaluate(statement2, env)
    elif keyword == "and":
        return And(tree[1:], env).eval()
    elif keyword == "or":
        return Or(tree[1:], env).eval()
    elif keyword == "del":
        _, variable = tree

        return env.delete(variable)

    elif keyword == "let":
        _, variable_bindings, body = tree

        # create a local env
        local_env = Environment({}, env)

        # evaluate var definitions and bind them to local env
        for var, val in variable_bindings:
            evaluated_val = evaluate(val, env)

            local_env.assign(var, evaluated_val)

        return evaluate(body, local_env)
    elif keyword == "set!":
        _, variable, expression = tree
        evaluated_val = evaluate(expression, env)

        env.set(variable, evaluated_val)

        return evaluated_val

    # evaluate each expression in the tree
    evaluated_expressions = [evaluate(expression, env) for expression in tree]

    # check if first evaluated expression is a function
    func = evaluated_expressions[0]

    # check if it's a CarlaeFunction
    if isinstance(func, CarlaeFunction):
        # print(evaluated_expressions)

        if len(evaluated_expressions) == 1:
            # function with no arguments
            return func.call([])

        # get the args
        args = evaluated_expressions[1:]

        # call the function on the args
        return func.call(args)

    if callable(func):
        # call the func on rest of evaluated expressions
        evaluated = func(evaluated_expressions[1:])

        return evaluated

    # if length of evaluated expressions is 1 return that value
    if len(evaluated_expressions) == 1:
        return evaluated_expressions[0]

    raise CarlaeEvaluationError()


def result_and_env(tree, env=None):
    # initialize environment for evaluation
    current_env = _create_env(env)
    evaluated = evaluate(tree, current_env)

    return evaluated, current_env


def run_carlae(raw_carlae_str, env=None):
    tokens = tokenize(raw_carlae_str)
    parsed = parse(tokens)
    evaluated = result_and_env(parsed, env)

    return evaluated[0]


def run_repl(env=None):
    env = _create_env(env)

    while True:
        raw_carlae_str = input("in> ")

        if raw_carlae_str == "EXIT":
            break

        try:
            value = run_carlae(raw_carlae_str, env)

            if _is_list(value):
                print(f"out> {render_list(value)}")

            else:
                print(f"out> {value}")

            # print(global_env)

        except Exception as e:
            exception_name = e.__class__.__name__

            print(exception_name)


def evaluate_file(filename, env=None):
    env = _create_env(env)

    with open(filename, "r") as f:
        carlae = f.read()

        return run_carlae(carlae, env)


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()

    builtins = Environment(carlae_builtins)
    global_env = Environment({}, builtins)

    files_to_load = sys.argv[1:]

    # load files
    for file_ in files_to_load:
        evaluate_file(file_, global_env)

    run_repl(global_env)

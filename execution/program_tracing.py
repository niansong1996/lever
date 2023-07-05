import os

from concurrent.futures import ProcessPoolExecutor as Pool
from typing import List, Dict, Tuple, Any, Union, NamedTuple, Set

from typing import List, Dict, Any
from execution.safe_execution_util import execute, canonicalize_var_dict, simple_canonicalize_var_dict
from tree_sitter import Language, Parser

ProgState = Dict[str, float]
HashableProgState = Set[float]
ProgTraceUnit = NamedTuple("ProgTraceUnit", [("code", str), ("type", str), ("state", ProgState)])
ProgTrace = List[ProgTraceUnit]
Program = NamedTuple("Program", [("code", str), ("code_lite", str), ("trace", ProgTrace)])

# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__)+'/../preprocessing/', 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

COM_STMTS = ['if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement',
             'function_definition', 'class_definition']
PY_MODULES = ['module', 'block', 'decorated_definition']

def byte_idx_to_char_idx(byte_idx: int, line: str) -> int:
    """convert byte index to char index"""
    return len(bytes(line, 'utf-8')[:byte_idx].decode('utf-8'))

def get_statements_from_code(code: str, parser, tolerate_errors: bool=False) -> List[Dict[str, Any]]:
    parsed_tree = parser.parse(bytes(code, 'utf-8'))

    # do a dfs on the parsed tree to record all the simple statements
    target_stmts: List[Dict] = []
    node_stack = [parsed_tree.root_node]
    while len(node_stack) > 0:
        node = node_stack.pop()

        if (node.type.endswith('statement') or node.type in ['comment', 'decorator']) \
            and node.type not in COM_STMTS:
            # this is a simple statement or a comment, so we can add it to the list
            target_stmts.append({'type': node.type, 'start_point': node.start_point, 
                                 'end_point': node.end_point, 'start_byte': node.start_byte, 
                                 'end_byte': node.end_byte})
        elif node.type in COM_STMTS or node.type.endswith('clause'):
            # separate the header and the body by the ":" token
            children_types = [c.type for c in node.children]
            separator_idx = children_types.index(':')
            assert separator_idx != -1

            # start of the header is the starter of the complex stmt, end is the end of the ":" token
            target_stmts.append({'type': node.type+'_header', 'start_point': node.start_point, 
                                 'start_byte': node.children[separator_idx].start_byte,
                                 'end_point': node.children[separator_idx].end_point, 
                                 'end_byte': node.children[separator_idx].end_byte})
            node_stack.extend(node.children[separator_idx+1:][::-1])
        elif node.type in PY_MODULES:
            node_stack.extend(node.children[::-1])
        elif node.type == 'ERROR':
            # err_code_line = code[:byte_idx_to_char_idx(node.end_byte, code)].split('\n')[-1]
            # print(f"failed to parse code: #########\n{err_code_line}\n#########")
            if tolerate_errors:
                continue
            else:
                # failed to parse tree, return None NOTE: previously returning [], but this will get 
                # confused with blank cells
                return None
        else:
            # other types, not sure what it contains, but assume it doesn't contain more statements
            print(f'unexpected node type: {node.type}')
            assert 'statement' not in node.sexp()

    return target_stmts

"""
Tracing the execution of a program:
    1. It parses the program into a sequence of tracing units (currently stmts);
    2. Make some markings of the tracing units;
    3. Insert tracing code to the program, after every tracing unit;
    4. Run the program with tracing;
    5. Collect the variable tracing information.
"""

from copy import deepcopy
from types import ModuleType

tracing_local_list = []
def record_state(local_var_dict):
    copied_local_var_dict = simple_canonicalize_var_dict(local_var_dict)
    tracing_local_list.append(copied_local_var_dict)

def get_function_final_state(program: str) -> Dict[str, Any]:
    # first parse the program with tree-sitter
    stmts = get_statements_from_code(program, parser)

    if stmts is None:
        return {"result": "ERROR: unparseable"}
    
    # put a record state before every return point
    program_with_tracing = ""
    program_bytes = bytes(program, "utf-8")
    byte_idx = 0
    for stmt in stmts:
        stmt_str = program_bytes[byte_idx:stmt['end_byte']+1].decode("utf-8")
        if stmt["type"] == "return_statement":
            # build the harness code
            return_token_idx = stmt_str.find("return")
            return_val_expr = stmt_str[return_token_idx:].replace("return", "").strip().strip(";")
            
            if len(return_val_expr) > 0:
                harness_code = f"_return_val={return_val_expr}; record_state(locals()); "

                # insert into the original return stmt
                # stmt_str = stmt_str[:return_token_idx] + harness_code + \
                #             stmt_str[return_token_idx:].replace(return_val_expr, " _return_val")
                stmt_str = stmt_str[:return_token_idx] + harness_code + "return _return_val\n"
            else:
                harness_code = f"record_state(locals()); "
                stmt_str = stmt_str[:return_token_idx] + harness_code + stmt_str[return_token_idx:]
        
        program_with_tracing += stmt_str
        byte_idx = stmt['end_byte']+1


    # execute the program with tracing code
    tracing_result = execute(program_with_tracing, {}, globals={
                              "tracing_local_list": tracing_local_list,
                              "record_state": record_state,
                              }, use_tracing=True, timeout=10, output_locals=False)
    
    return tracing_result

def assertion_to_test(assertion: str) -> str:
    """ get rid of the expected results in the assertion """
    program_bytes = bytes(assertion, 'utf-8')
    parsed_tree = parser.parse(program_bytes)

    root_node = parsed_tree.root_node
    assert len(root_node.children) == 1

    assert_stmt = root_node.children[0]
    assert assert_stmt.type == "assert_statement"
    # assert len(assert_stmt.children) == 2 # NOTE: it might break if something like "assert a == b,c"

    comparison_stmt = assert_stmt.children[1]
    assert comparison_stmt.type == "comparison_operator"
    assert len(comparison_stmt.children) == 3

    call_stmt = comparison_stmt.children[0]
    while call_stmt.type == "parenthesized_expression":
        assert len(call_stmt.children) == 3
        call_stmt = call_stmt.children[1]
    assert call_stmt.type == "call"

    call_str = program_bytes[call_stmt.start_byte:call_stmt.end_byte].decode("utf-8").strip()

    return call_str
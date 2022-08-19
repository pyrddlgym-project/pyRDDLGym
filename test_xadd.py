# from xaddpy.xadd import xadd_parse_utils
from xaddpy.xadd.xadd import XADD
import sympy as sp

def test_xadd():
    context = XADD()
    x, y = sp.S('x'), sp.S('y')
    """
    Create a node 
        ([x - y - 5 <= 0]
            ([x ** 2])              # when the decision expression holds true
            ([10])                  # otherwise
        )
    """
    dec_expr1 = x - y <= 5

    xadd_as_list1 = [dec_expr1, [x ** 2], [sp.S(10)]]  # constant numbers should be passed through sympy.S()
    node1: int = context.build_initial_xadd(
        xadd_as_list1)  # This method recursively builds an XADD node given a nested list of expressions
    print(f"Node 1:\n{context.get_exist_node(node1)}")
    """
    Create another node 
        ([x + 2 * y <= 0]
            ([-2 * y])              # when the decision expression holds true
            ([3 * x])               # otherwise
        )
    """
    dec_expr2 = x + 2 * y <= 0
    dec_id2, is_reversed = context.get_dec_expr_index(dec_expr2, create=True)
    high: int = context.get_leaf_node(sp.S(- 2) * y)  # You can instantiate a leaf node by passing the expression
    low: int = context.get_leaf_node(sp.S(3) * x)
    if is_reversed:  # In case the canonical expression associated with `dec_id` is reversed,
        tmp = low;
        low = high;
        high = tmp  # swap low and high
    node2: int = context.get_internal_node(dec_id=dec_id2, low=low, high=high)
    print(f"Node 2:\n{context.get_exist_node(node2)}")

    # Examples of some basic operations between the two XADDs
    node_sum = context.apply(node1, node2, op='sum')
    print(f"sum :\n{context.get_exist_node(node_sum)}")
    node_prod = context.apply(node1, node2, op='prod')
    print(f"prod :\n{context.get_exist_node(node_prod)}")
    node_case_min = context.apply(node1, node2, op='min')
    print(f"min :\n{context.get_exist_node(node_case_min)}")
    node_case_max = context.apply(node1, node2, op='max')
    print(f"max :\n{context.get_exist_node(node_case_max)}")



    """
    Additional notes: 
        this repo selectively implemented necessary components from the original Java XADD code.
        So, there should be some missing functionalities and some operations may not be supported in the current form.
    """
    return


if __name__ == "__main__":
    test_xadd()
# This file is based on thiago pbueno's pyrddl.
# https://github.com/thiagopbueno/pyrddl
# it was adapted and extended for pyRDDLGym

def rename_next_state_fluent(name: str) -> str:
    '''Returns next state fluent canonical name.
    Args:
        name (str): The current state fluent name.
    Returns:
        str: The next state fluent name.
    '''
    i = name.index('/')
    functor = name[:i-1]
    arity = name[i+1:]
    return "{}/{}".format(functor, arity)


def rename_state_fluent(name: str) -> str:
    '''Returns current state fluent canonical name.
    Args:
        name (str): The next state fluent name.
    Returns:
        str: The current state fluent name.
    '''
    i = name.index('/')
    functor = name[:i]
    arity = name[i+1:]
    return "{}'/{}".format(functor, arity)

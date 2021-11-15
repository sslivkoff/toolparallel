import functools
import importlib
import inspect
import types

from . import parallel_maps


def parallelize_input(
    singular_arg,
    plural_arg,
    pass_f_by_ref=True,
    config=None,
):
    """decorator to create parallelized version of function

    - parallel mode of function is activated when plural_arg is passed
    """

    if config is None:
        config = {}

    def decorator(f):

        # check that `blocks` is an arg and `blocks` is not an arg
        argspec = inspect.getfullargspec(f)
        argnames = argspec.args + argspec.kwonlyargs
        if singular_arg not in argnames:
            raise Exception('function does not use singular parameter')
        if plural_arg in argnames:
            raise Exception('function already uses plural parameter')
        if 'parallel_kwargs' in argnames:
            raise Exception('function already uses parallel_kwargs')

        @functools.wraps(f)
        def wrapped_f(*args, parallel_kwargs=None, **kwargs):

            if parallel_kwargs is None:
                parallel_kwargs = {}
            parallel_kwargs = dict(config, **parallel_kwargs)

            if plural_arg in kwargs and kwargs[plural_arg] is not None:
                values = kwargs[plural_arg]
                common = {k: v for k, v in kwargs.items() if k != plural_arg and k != singular_arg}
                common.update(dict(zip(argspec.args, args)))

                if pass_f_by_ref:
                    f_ref = (f.__module__, f.__name__)
                else:
                    f_ref = f

                return parallel_maps.parallel_map(
                    f=functools.partial(_f_execute, f_ref=f_ref),
                    arg_list=values,
                    arg_name=singular_arg,
                    common=common,
                    **parallel_kwargs
                )

            else:
                if plural_arg in kwargs:
                    kwargs.pop(plural_arg)
                return f(*args, **kwargs)

        return wrapped_f

    return decorator


def _f_execute(*args, f_ref, **kwargs):
    if isinstance(f_ref, tuple):
        module_name, f_name = f_ref
        module = importlib.import_module(module_name)
        f = getattr(module, f_name)
    elif isinstance(f_ref, types.FunctionType):
        f = f_ref
    else:
        raise Exception()

    return f(*args, **kwargs)


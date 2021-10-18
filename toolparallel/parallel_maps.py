import concurrent.futures
import functools
import multiprocessing.pool


def parallel_map(
    f,
    n_workers=None,
    mode=None,
    arg_list=None,
    arg_name=None,
    arg_lists=None,
    kwarg_dicts=None,
    common=None,
    to_dict_of_lists=False,
    n_subworkers_per_worker=None,
):
    """compute f for each input args, parallelizing with processes or threads

    ## Parallel Modes
    - process: use a pool of processes to compute map
    - thread: use a pool of threads to compute map
    - serial: do not use parallelization
    - nested: use multiple processes each of which uses multiple threads

    ## Single argument will be one of:
    - arg_list + named_arg -> single keyword arg
    - arg_list - named_arg -> single positional arg
    - arg_lists -> a list of positional args
    - kwarg_dicts -> a dict of kwargs

    ## Inputs
    - f: function used to apply map
    - n_workers: number of workers
    - mode: str of parallelization mode (see above)
    - arg_list: list of single args, or dict of {index_value: arg}
    - arg_lists: list of `*args` lists, or dict of {index_value: args}
    - kwarg_dicts: list of `**kwargs` lists, or dict of {index_value: kwargs}
    - common: dict of kwargs that are passed to each call of f
    - to_dict_of_lists: bool to collapse list of dicts output to dict of lists

    ## Output
    - if inputs is a list: return list of outputs
    - if inputs is a dict: return dict with same keys, and values as outputs
    - if to_dict_of_lists: return dict of lists instead of list of dicts
        - not implemented for indexed inputs
    """

    if mode is None:
        mode = 'process'
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if n_subworkers_per_worker is not None and mode != 'nested':
        raise Exception('n_subworkers_per_worker only valid for nested mode')

    # create wrapped f and inputs
    wrapped_f, inputs, index = _wrap_f(
        f=f,
        arg_list=arg_list,
        arg_name=arg_name,
        arg_lists=arg_lists,
        kwarg_dicts=kwarg_dicts,
        common=common,
    )
    if to_dict_of_lists and index is not None:
        raise NotImplementedError('collapsing dict keys with indexed inputs')

    # compute map
    if mode == 'process':
        pool = multiprocessing.pool.Pool(n_workers)
        output = pool.map(wrapped_f, inputs)
    elif mode == 'thread':
        executor = concurrent.futures.ThreadPoolExecutor(n_workers)
        result = executor.map(wrapped_f, inputs)
        output = list(result)
    elif mode == 'serial':
        output = [wrapped_f(arg) for arg in inputs]
    elif mode == 'nested':
        raise NotImplementedError()
        output = []
    else:
        raise Exception('unknown mode: ' + str(mode))

    # collapse list of dicts into dict of lists
    if to_dict_of_lists and len(output) > 0:
        output = _list_of_dicts_to_dict_of_lists(output, deep=True)

    # reapply index
    if index is not None:
        output = dict(zip(index, output))

    return output


#
# # helper functions
#


def _f_arg(arg, f, common):
    return f(arg, **common)


def _f_named_arg(arg, f, common, name):
    return f(**{name: arg}, **common)


def _f_args(args, f, common):
    return f(*args, **common)


def _f_kwargs(kwargs, f, common):
    return f(**kwargs, **common)


# def _wrapped_f(payload, f, common, arg_mode, name=None, verbose=True, print_every=100):

#     # print preamble
#     if verbose:
#         _, job_index, n_jobs = payload
#         if job_index % print_every == 0:
#             width = len(str(n_jobs))
#             format = '%' + str(width) + 'd'
#             print(format % job_index, '/', n_jobs)

#     # call function
#     if arg_mode == 'arg':
#         arg = payload[0]
#         return f(arg, **common)
#     elif arg_mode == 'named_arg':
#         arg = payload[0]
#         return f(**{name: arg}, **common)
#     elif arg_mode == 'args':
#         args = payload[0]
#         return f(*args, **common)
#     elif arg_mode == 'kwargs':
#         kwargs = payload[0]
#         return f(**kwargs, **common)
#     else:
#         raise Exception('unknown arg_mode: ' + str(arg_mode))


def _wrap_f(f, arg_list, arg_lists, arg_name, kwarg_dicts, common):
    """wrap f to accept a single argument and return list of single arg inputs
    """

    # validate inputs
    if common is None:
        common = {}
    if [arg_list, arg_lists, kwarg_dicts].count(None) != 2:
        raise Exception('must specify arg_list, arg_lists, or kwarg_dicts')

    # create outputs
    partial_kwargs = {'f': f, 'common': common}
    if arg_list is not None and arg_name is None:
        inputs = arg_list
        wrapped_f = functools.partial(_f_arg, **partial_kwargs)
    elif arg_list is not None and arg_name is not None:
        inputs = arg_list
        wrapped_f = functools.partial(
            _f_named_arg, name=arg_name, **partial_kwargs
        )
    elif arg_lists is not None:
        inputs = arg_lists
        wrapped_f = functools.partial(_f_args, **partial_kwargs)
    elif kwarg_dicts is not None:
        inputs = kwarg_dicts
        wrapped_f = functools.partial(_f_kwargs, **partial_kwargs)
    else:
        raise Exception('must specify arg_list, arg_lists, or kwarg_dicts')

    # create index
    if isinstance(inputs, dict):
        index = list(inputs.keys())
        inputs = list(inputs.values())
    else:
        index = None

    # payloads = [[item, i, len(inputs)] for i, item in enumerate(inputs)]

    return wrapped_f, inputs, index


def _list_of_dicts_to_dict_of_lists(list_of_dicts, deep=False):
    """

    currently does very rough check to see which keys should be converted deeply
    """

    # collect keys for final output
    if not isinstance(list_of_dicts[0], dict):
        raise Exception('outputs are not of type dict')
    keys = set(list_of_dicts[0].keys())
    dict_of_lists = {key: [] for key in keys}

    # main conversion
    for dict_item in list_of_dicts:
        if not isinstance(dict_item, dict):
            raise Exception('outputs are not of type dict')
        if set(dict_item.keys()) != set(keys):
            raise Exception('different keys across outputs')
        for key, value in dict_item.items():
            dict_of_lists[key].append(value)

    # recursive conversions
    if deep:

        # check which entries are lists of dicts
        deep_keys = []
        for key, list_items in dict_of_lists.items():
            if all(isinstance(item, dict) for item in list_items):
                deep_keys.append(key)

        for deep_key in deep_keys:
            dict_of_lists[deep_key] = _list_of_dicts_to_dict_of_lists(
                dict_of_lists[deep_key],
                deep=deep,
            )

    return dict_of_lists


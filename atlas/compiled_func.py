""" Namespace for compiled funcs """

from .operators import OpInfo

#def _atlas_compiled_function_1(name, *, _atlas_gen_exec_env=None, _atlas_gen_strategy=None, _atlas_gen_hooks=None):
#    _op_info_0 = OpInfo(sid='/initial_extractor/Select@@1', gen_name='initial_extractor', op_type='Select',
#                        index=1, gen_group=None, uid=None, tags=None)
#    _op_info_1 = OpInfo(sid='/initial_extractor/Select@@2', gen_name='initial_extractor', op_type='Select',
#                        index=2, gen_group=None, uid=None, tags=None)
#
#    _handler_0 = _atlas_gen_strategy.get_op_handler(_op_info_0)
#    _handler_1 = _atlas_gen_strategy.get_op_handler(_op_info_1)
#
#    first = _atlas_gen_strategy.generic_call(name, op_info=_op_info_0, handler=_handler_0)
#    last = _atlas_gen_strategy.generic_call(name, op_info=_op_info_1, handler=_handler_1)
#    return f'{first}.{last}'
#
#
#def _atlas_compiled_function_1(name, *, _atlas_gen_exec_env=None, _atlas_gen_strategy=None, _atlas_gen_hooks=None):
#    first = _atlas_hook_wrapper(name, op_info=_op_info_1, handler=_handler_0, _atlas_gen_hooks=_atlas_gen_hooks, _atlas_gen_strategy=_atlas_gen_strategy)
#    last = _atlas_hook_wrapper(name, op_info=_op_info_2, handler=_handler_1, _atlas_gen_hooks=_atlas_gen_hooks, _atlas_gen_strategy=_atlas_gen_strategy)
#    return f'{first}.{last}'
#

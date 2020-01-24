import torch
import collections


def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)

# # theano-like functions
#
# def function(inputs, outputs, updates=None, givens=None):
#     """Just like Theano function. Take a bunch of tensorflow placeholders and expersions
#     computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
#     values to be feed to the inputs placeholders and produces the values of the experessions
#     in outputs.
#
#     Input values can be passed in the same order as inputs or can be provided as kwargs based
#     on placeholder name (passed to constructor or accessible via placeholder.op.name).
#
#     Example:
#         x = tf.placeholder(tf.int32, (), name="x")
#         y = tf.placeholder(tf.int32, (), name="y")
#         z = 3 * x + 2 * y
#         lin = function([x, y], z, givens={y: 0})
#
#         with single_threaded_session():
#             initialize()
#
#             assert lin(2) == 6
#             assert lin(x=3) == 9
#             assert lin(2, 2) == 10
#             assert lin(x=2, y=3) == 12
#
#     Parameters
#     ----------
#     inputs: [tf.placeholder or TfInput]
#         list of input arguments
#     outputs: [tf.Variable] or tf.Variable
#         list of outputs or a single output to be returned from function. Returned
#         value will also have the same shape.
#     """
#     if isinstance(outputs, list):
#         return _Function(inputs, outputs, updates, givens=givens)
#     elif isinstance(outputs, (dict, collections.OrderedDict)):
#         f = _Function(inputs, outputs.values(), updates, givens=givens)
#         return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
#     else:
#         f = _Function(inputs, [outputs], updates, givens=givens)
#         return lambda *args, **kwargs: f(*args, **kwargs)[0]
#
#
# class _Function(object):
#     def __init__(self, inputs, outputs, updates, givens, check_nan=False):
#         for inpt in inputs:
#             if not issubclass(type(inpt), TfInput):
#                 assert len(inpt.op.inputs) == 0, "inputs should all be placeholders of rl_algs.common.TfInput"
#         self.inputs = inputs
#         updates = updates or []
#         self.update_group = tf.group(*updates)
#         self.outputs_update = list(outputs) + [self.update_group]
#         self.givens = {} if givens is None else givens
#         self.check_nan = check_nan
#
#     def _feed_input(self, feed_dict, inpt, value):
#         if issubclass(type(inpt), TfInput):
#             feed_dict.update(inpt.make_feed_dict(value))
#         elif is_placeholder(inpt):
#             feed_dict[inpt] = value
#
#     def __call__(self, *args, **kwargs):
#         assert len(args) <= len(self.inputs), "Too many arguments provided"
#         feed_dict = {}
#         # Update the args
#         for inpt, value in zip(self.inputs, args):
#             self._feed_input(feed_dict, inpt, value)
#         # Update the kwargs
#         kwargs_passed_inpt_names = set()
#         for inpt in self.inputs[len(args):]:
#             inpt_name = inpt.name.split(':')[0]
#             inpt_name = inpt_name.split('/')[-1]
#             assert inpt_name not in kwargs_passed_inpt_names, \
#                 "this function has two arguments with the same name \"{}\", so kwargs cannot be used.".format(inpt_name)
#             if inpt_name in kwargs:
#                 kwargs_passed_inpt_names.add(inpt_name)
#                 self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
#             else:
#                 assert inpt in self.givens, "Missing argument " + inpt_name
#         assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
#         # Update feed dict with givens.
#         for inpt in self.givens:
#             feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
#         results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
#         if self.check_nan:
#             if any(np.isnan(r).any() for r in results):
#                 raise RuntimeError("Nan detected")
#         return results

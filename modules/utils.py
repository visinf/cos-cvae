import torch
import torch.nn


def tiny_value_of_dtype(dtype: torch.dtype):
	"""
	Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
	issues such as division by zero.
	This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
	Only supports floating point dtypes.
	"""
	if not dtype.is_floating_point:
		raise TypeError("Only supports floating point dtypes.")
	if dtype == torch.float or dtype == torch.double:
		return 1e-13
	elif dtype == torch.half:
		return 1e-4
	else:
		raise TypeError("Does not support dtype " + str(dtype))

def masked_softmax(
	vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1, memory_efficient: bool = False,
) -> torch.Tensor:
	"""
	`torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
	masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
	`None` in for the mask is also acceptable; you'll just get a regular softmax.
	`vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
	broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
	unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
	do it yourself before passing the mask into this function.
	If `memory_efficient` is set to true, we will simply use a very large negative number for those
	masked positions so that the probabilities of those positions would be approximately 0.
	This is not accurate in math, but works for most cases and consumes less memory.
	In the case that the input vector is completely masked and `memory_efficient` is false, this function
	returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
	a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
	will treat every element as equal, and do softmax over equal numbers.
	"""
	if mask is None:
		result = torch.nn.functional.softmax(vector, dim=dim)
	else:
		while mask.dim() < vector.dim():
			mask = mask.unsqueeze(1)
		if not memory_efficient:
			# To limit numerical errors from large vector elements outside the mask, we zero these out.
			result = torch.nn.functional.softmax(vector * mask.float(), dim=dim)
			result = result * mask.float()
			result = result / (
				result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
			)
		else:
			masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
			result = torch.nn.functional.softmax(masked_vector, dim=dim)
	return result


def masked_mean( vector, mask, dim, keepdim = False):
	replaced_vector = vector.masked_fill(~mask, 0.0)

	value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
	value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
	return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def sequence_cross_entropy_with_logits( logits, targets, weights, average = "batch", label_smoothing = None, gamma = None, alpha = None):
	"""
	Computes the cross entropy loss of a sequence, weighted with respect to
	some user provided weights. Note that the weighting here is not the same as
	in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
	classes; here we are weighting the loss contribution from particular elements
	in the sequence. This allows loss computations for models which use padding.

	Parameters
	----------
	logits : ``torch.FloatTensor``, required.
		A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
		which contains the unnormalized probability for each class.
	targets : ``torch.LongTensor``, required.
		A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
		index of the true class for each corresponding step.
	weights : ``torch.FloatTensor``, required.
		A ``torch.FloatTensor`` of size (batch, sequence_length)
	average: str, optional (default = "batch")
		If "batch", average the loss across the batches. If "token", average
		the loss across each item in the input. If ``None``, return a vector
		of losses per batch element.
	label_smoothing : ``float``, optional (default = None)
		Whether or not to apply label smoothing to the cross-entropy loss.
		For example, with a label smoothing value of 0.2, a 4 class classification
		target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
		the correct label.
	gamma : ``float``, optional (default = None)
		Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
		well-classified examples and put more focus on hard. The greater value
		``gamma`` is, the more focus on hard examples.
	alpha : ``float`` or ``List[float]``, optional (default = None)
		Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
		used independently with ``gamma``. If a single ``float`` is provided, it
		is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
		negative respectively. If a list of ``float`` is provided, with the same
		length as the number of classes, the weights will match the classes.
		[*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
		Dense Object Detection," 2017 IEEE International Conference on Computer
		Vision (ICCV), Venice, 2017, pp. 2999-3007.

	Returns
	-------
	A torch.FloatTensor representing the cross entropy loss.
	If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
	If ``average is None``, the returned loss is a vector of shape (batch_size,).

	"""
	if average not in {None, "token", "batch"}:
		raise ValueError("Got average f{average}, expected one of "
						 "None, 'token', or 'batch'")

	# make sure weights are float
	weights = weights.float()
	# sum all dim except batch
	non_batch_dims = tuple(range(1, len(weights.shape)))
	# shape : (batch_size,)
	weights_batch_sum = weights.sum(dim=non_batch_dims)
	# shape : (batch * sequence_length, num_classes)
	logits_flat = logits.view(-1, logits.size(-1))
	# shape : (batch * sequence_length, num_classes)
	log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
	# shape : (batch * max_len, 1)
	targets_flat = targets.view(-1, 1).long()
	# focal loss coefficient
	if gamma:
		# shape : (batch * sequence_length, num_classes)
		probs_flat = log_probs_flat.exp()
		# shape : (batch * sequence_length,)
		probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
		# shape : (batch * sequence_length,)
		focal_factor = (1. - probs_flat) ** gamma
		# shape : (batch, sequence_length)
		focal_factor = focal_factor.view(*targets.size())
		weights = weights * focal_factor

	if alpha is not None:
		# shape : () / (num_classes,)
		if isinstance(alpha, (float, int)):
			# pylint: disable=not-callable
			# shape : (2,)
			alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
										dtype=weights.dtype, device=weights.device)
			# pylint: enable=not-callable
		elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
			# pylint: disable=not-callable
			# shape : (c,)
			alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
			# pylint: enable=not-callable
			if not alpha_factor.size():
				# shape : (1,)
				alpha_factor = alpha_factor.view(1)
				# shape : (2,)
				alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
		else:
			raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
							 '{} provided.').format(type(alpha)))
		# shape : (batch, max_len)
		alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
		weights = weights * alpha_factor

	if label_smoothing is not None and label_smoothing > 0.0:
		num_classes = logits.size(-1)
		smoothing_value = label_smoothing / num_classes
		# Fill all the correct indices with 1 - smoothing value.
		one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
		smoothed_targets = one_hot_targets + smoothing_value
		negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
		negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
	else:
		# Contribution to the negative log likelihood only comes from the exact indices
		# of the targets, as the target distributions are one-hot. Here we use torch.gather
		# to extract the indices of the num_classes dimension which contribute to the loss.
		# shape : (batch * sequence_length, 1)
		#print('log_probs_flat ',log_probs_flat.size(),' targets_flat ',targets_flat.size())
		negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
	# shape : (batch, sequence_length)
	negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
	# shape : (batch, sequence_length)
	negative_log_likelihood = negative_log_likelihood * weights

	if average == "batch":
		# shape : (batch_size,)
		per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
		num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
		return per_batch_loss.sum() / num_non_empty_sequences
	elif average == "token":
		return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
	else:
		# shape : (batch_size,)
		per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
		return per_batch_loss	
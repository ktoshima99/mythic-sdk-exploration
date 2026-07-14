"""Generate randomized hardware model instances for Monte Carlo based evaluation."""
import math

from funcy import omit

from munc._constants import ONNXType


def freeze_hardware_parameters(sess, hw_parameters, initializers):
    """Monkeypatch Session to reuse provided hardware parameters and initializers."""
    orig_make_torch_net = sess.make_torch_net

    def make_torch_net_fixed():
        torch_model = orig_make_torch_net()
        hw_models = torch_model.get_hardware_models()
        for m in hw_models.values():
            m.update_models_every_time = False
        torch_model.forward_random_inputs(sess.model)
        for k, v in hw_models.items():
            v.set_hardware_parameters(hw_parameters[k])
        return torch_model

    sess.make_torch_net = make_torch_net_fixed
    sess.saved_initializers = {name: sess.model.get_initializer_np(name) for name in initializers.keys()}
    for name, value in initializers.items():
        sess.model.set_initializer_np(name, value)


def unfreeze_hardware_parameters(sess):
    """Restore the original Session.make_torch_net."""
    del sess.make_torch_net
    for name, value in sess.saved_initializers.items():
        sess.model.set_initializer_np(name, value)
    del sess.saved_initializers


def get_schedule_num_samples(schedule):
    """Return the total number of instances defined by `schedule`."""
    return math.prod(step['repeat'] for step in schedule)


def random_model_instances(sess, schedule, num_tests=None):
    """Yield Sessions with frozen hardware parameters for each step of the schedule.

    Parameters
    ----------
    sess : Session
        Session whose torch nets are generated and frozen.
    schedule : List[Dict]
        Sequence of schedule entries. Each entry must include a 'repeat' key that indicates how many times to reuse the
        associated nonideality settings. Hardware models are randomized when advancing to a new schedule state and
        frozen for the yielded session.
    num_tests : Optional[int]
        Number of sessions to generate. Defaults to the total steps across the schedule.

    Yields
    ------
    Session
        The input session with make_torch_net patched to return a torch model whose hardware parameters are fixed for
        that iteration.
    """
    if num_tests is None:
        num_tests = get_schedule_num_samples(schedule)

    torch_model = sess.make_torch_net()
    hw_models = torch_model.get_hardware_models()
    for m in hw_models.values():
        m.update_models_every_time = False
    torch_model.forward_random_inputs(sess.model)

    for i in range(num_tests):
        _randomize_model(torch_model, schedule, i - 1, i)
        initializers = _randomize_weights(sess.model, schedule, i - 1, i)
        hw_parameters = {k: v.get_hardware_parameters() for k, v in hw_models.items()}
        freeze_hardware_parameters(sess, hw_parameters, initializers)
        try:
            yield sess
        finally:
            unfreeze_hardware_parameters(sess)


def _get_schedule_state(schedule, i):
    """Decode a flat index into per-step positions for the provided schedule."""
    indices = []
    for step in reversed(schedule):
        repeat = step['repeat']
        assert repeat >= 2
        indices.append(i % repeat)
        i = i // repeat
    indices.reverse()
    return indices


def _randomize_model(torch_model, schedule, current_idx, new_idx):
    """Reconfigure hardware models when the schedule state changes between indices."""
    hw_models = torch_model.get_hardware_models()
    current_schedule_state = _get_schedule_state(schedule, current_idx)
    new_schedule_state = _get_schedule_state(schedule, new_idx)
    for curr, new, schedule_step in zip(current_schedule_state, new_schedule_state, schedule):
        if curr != new:
            for hw_model in hw_models.values():
                hw_model.configure_nonidealities(**omit(schedule_step, ('repeat', 'weight_randomizer')))


def _randomize_weights(onnx_model, schedule, current_idx, new_idx):
    """Reconfigure hardware models when the schedule state changes between indices."""
    current_schedule_state = _get_schedule_state(schedule, current_idx)
    new_schedule_state = _get_schedule_state(schedule, new_idx)
    weight_randomizers = [schedule_step.get('weight_randomizer') for curr, new, schedule_step
                          in zip(current_schedule_state, new_schedule_state, schedule)
                          if curr != new and schedule_step.get('weight_randomizer', False)]
    initializers = {}
    if weight_randomizers:
        assert len(weight_randomizers) == 1, "Don't know how to apply multiple weight randomizers. It's a new use case."
        weight_randomizer = weight_randomizers[0]
        for mma in onnx_model.get_nodes_with_op_type([ONNXType.MYTHIC_CONV, ONNXType.MYTHIC_LINEAR]):
            initializers.update(weight_randomizer(mma))
    return initializers

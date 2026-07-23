# noqa-flake8-docstrings
import logging
import copy

logger = logging.getLogger('BCM')


class _Factory:

    def __init__(self):
        self._builders = {}

    def register_builder(self, name, builder):
        self._builders[name] = builder

    def __getitem__(self, name):
        try:
            return self._builders[name]
        except ValueError:
            raise ValueError(
                f"No builder with name {name} is registered",
                f"Available builders are {self._builder}"
            )

    def __getattr__(self, name):
        return self[name]


class _MMAFactory(_Factory):
    """Factory holding the different BCM MMAs.

    Example
    -------
    mma_attrs = MMAAttributeFactory['ideal']
    mma = MMA['ideal'](weights, biases, iFSR=2, pFSR=2, mma_attributes=mma_attrs)
    mma.dot(x)

    or

    mma_attrs = MMAAttributeFactory.ideal
    mma = MMA.ideal(weights, biases, iFSR=2, pFSR=2, mma_attributes=mma_attrs)
    mma.dot(x)
    """

    def get_available_models(self):
        return list(self._builders.keys())

    def create_mma(self, str_mma, weights, biases, pFSR=1, iFSR=1):
        try:
            mma_attr = MMAAttributeFactory[str_mma]()
            mma = self[str_mma](weights, biases, mma_attr=mma_attr, pFSR=pFSR, iFSR=iFSR)
        except KeyError:
            raise KeyError(f'Unknown MMA type {str_mma}')
        return mma


class _MMAAttributeFactory(_Factory):

    def from_string(self, attr_name):
        """Return the mma attributes from a string key.

        Example
        -------
        attrs = MMAAttributeFactory.from_string('numpy.hw_accurate')
        """
        if '.' in attr_name:
            mma_type, preset_attrs = attr_name.split('.')
        else:
            mma_type = attr_name
            preset_attrs = None

        try:
            default_attrs = self[mma_type]
        except KeyError:
            raise KeyError(f"Unable to find defaults for MMA '{mma_type}'")

        if preset_attrs is None:
            attrs = default_attrs
        else:
            try:
                attrs_preset = getattr(default_attrs, preset_attrs)
            except AttributeError:
                raise AttributeError(f"Unable to find preset for MMA '{mma_type}'")
            else:
                attrs = attrs_preset()

        return attrs

    def load_config(self, attr_name, new_attrs):
        """Override attributes using values from a dictionary.

        Example
        -------
        attrs = MMAAttributeFactory.load_config('numpy', {'clip_outputs': True})
        """
        attrs = copy.deepcopy(self.from_string(attr_name))

        for key, val in new_attrs.items():
            if hasattr(attrs, key):
                old_val = getattr(attrs, key)
                setattr(attrs, key, val)
                logger.info(f"[{key}] {old_val} --> {val}")
            else:
                print(f"Attempting to set attribute {key}; however, this is not an available attribute")

        return attrs


MMA = _MMAFactory()
MMAAttributeFactory = MMAAttrributeFactory = _MMAAttributeFactory()


class SimpleAttributes:
    """SimpleAttributes class to hold the attributes of SimpleModel."""

    def __init__(self):
        # ADC noise
        self.simple_noise = 68e-9
        self.simple_offset = 23e-9
        self.simple_inl = -0.04
        # flash noise
        self.pop_lognorm_mean = -4.6
        self.pop_lognorm_sigma = 1.35
        self.pop_fraction = 1
        self.decay_rate = 0
        self.decay_hours = 0
        self.temp_delta = 5
        self.mc_mult = .1
        self.mc_mult_sigma_lsb = 0
        self.linear_beta0 = 0
        self.linear_beta1 = 1

    @classmethod
    def no_noise(cls):
        attrs = cls()
        # ADC noise
        attrs.simple_noise = 0
        attrs.simple_offset = 0
        attrs.simple_inl = 0
        # flash noise
        attrs.pop_lognorm_mean = 0
        attrs.pop_lognorm_sigma = 0
        attrs.pop_fraction = 0
        attrs.decay_rate = 0
        attrs.decay_hours = 0
        attrs.temp_delta = 0
        attrs.mc_mult = 0
        attrs.mc_mult_sigma_lsb = 0
        attrs.linear_beta0 = 0
        attrs.linear_beta1 = 1
        return attrs

# MMAAttributeFactory.register_builder("simple", SimpleAttributes())

import importlib
import importlib.util
import json
import pathlib
import prodict

import utils.constants as consts
from utils.schema import DefaultValidator, DEFAULT_OPTIONS_SCHEMA


class BadassOptions(prodict.Prodict):

    @classmethod
    def from_dict(cls, input_dict):
        # Override Prodict.from_dict to normalize and validate input
        v = DefaultValidator(DEFAULT_OPTIONS_SCHEMA)

        # Update dict with default values if needed
        input_dict = v.normalized(input_dict)
        if not v.validate(input_dict):
            raise Exception('Options validation failed: %s' % v.errors)

        return super().from_dict(input_dict)

    @classmethod
    def from_file(cls, _filepath):
        filepath = pathlib.Path(_filepath)
        if not filepath.exists():
            raise Exception('Unable to find options file: %s' % str(filepath))

        ext = filepath.suffix[1:]
        parse_func_name = 'parse_%s' % ext
        if not hasattr(cls, parse_func_name):
            raise Exception('Unsupported option file type: %s' % ext)

        return getattr(cls, parse_func_name)(filepath)


    # Custom file type parsers
    # Note: each parser should parse options to a dict and use
    #   BadassOptions.from_dict to initialize, allowing for
    #   option normalization and validation

    @classmethod
    def parse_json(cls, filepath):
        return cls.from_dict(json.load(filepath.open()))

    @classmethod
    def parse_py(cls, filepath):
        spec = importlib.util.spec_from_file_location('optmod', filepath)
        optmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optmod)
        return cls.from_dict({k:getattr(optmod, k) for k in dir(optmod) if not k[:2] == '__'})

    @classmethod
    def get_options(cls, options_data):
        if isinstance(options_data, list):
            return [cls.get_options(o) for o in options_data]

        if isinstance(options_data, dict):
            return [cls.from_dict(options_data)]

        if isinstance(options_data, pathlib.Path) or isinstance(options_data, str):
            return [cls.from_file(options_data)]

        return []

    @classmethod
    def get_options_dep(cls, args):
        # function to handle the traditional way to call run_BADASS

        options_file = args.get('options_file')
        ret = cls.parse_py(options_file)
        breakpoint()

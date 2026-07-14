# This file is distributed under the terms of Mythic Inc's Software Licence Agreement
# Copyright (C) 2021, Mythic Inc. All rights reserved.
#
"""
Setup script for building, distributing, and installing L0 IR Python functionality.
"""
import json
from setuptools import setup, find_namespace_packages


def main():
    name = 'mythic-l0'

    setup(
        name=name,
        use_scm_version={'root': '/opt/dfvm'},
        setup_requires=['setuptools_scm'],
        description='Mythic L0 IR formats and support tools',
        license='Proprietary',
        packages=find_namespace_packages(include=['mythic*']),
        install_requires=[
            'numpy',
            'mythic-target-spec >= 0.17.1',
        ],
    )


if __name__ == '__main__':
    main()

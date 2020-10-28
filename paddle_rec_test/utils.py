#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ruamel import yaml
import built_in
import subprocess


def get_yaml_content(file_name):
    """
    get content from yaml file
    Args:
        file_name(str):
    Returns:
        content(dict):
    """
    with open(file_name) as f:
        content = yaml.load(f, Loader=yaml.RoundTripLoader)

        # output: <type 'OrderedDict'>
        print(type(content))
        return content


def generate_yaml_file(content, file_name):
    """
    generate yaml file based on content.
    Args:
        content(dict):
        file_name(str):
    Returns:
    """
    with open(file_name, 'w') as nf:
        yaml.dump(content, nf, Dumper=yaml.RoundTripDumper)


def check_result(operate, expect_value, check_value):
    """
    check the result.
    Args:
        operate:
        expect:
        result:

    Returns:

    """
    if operate in built_in.__dict__:
        built_in.__dict__[operate](expect_value, check_value)
    else:
        raise Exception("not support this operate named: {}".format(operate))


def get_platform():
    """get running platform."""
    import platform
    plats = platform.platform()
    if 'Linux' in plats:
        return "LINUX"
    if 'Darwin' in plats:
        return "DARWIN"
    if 'Windows' in plats:
        return "WINDOWS"


def cmd_shell(cmd):
    """

    Args:
        cmd:

    Returns:
    """
    pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pro.wait()
    return pro.returncode

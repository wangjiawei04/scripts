#!/usr/bin/env python
# -*- coding: utf-8 -*-
from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.instance import SingleInstance
from paddlerec.core.trainers.framework.network import SingleNetwork
from paddlerec.core.trainers.framework.startup import SingleStartup
from paddlerec.core.trainers.framework.runner import SingleRunner
from paddlerec.core.trainers.framework.terminal import TerminalBase

class Instance(SingleInstance):
    """Instance
    """
    def __init__(self, context):
        # print("======== User Define SingleInstance, context is {}".format(context))
        print("======== User Define SingleInstance")
        pass

    def instance(self, context):
        """crete instance."""
        super(Instance, self).instance(context)


class Network(SingleNetwork):
    """Net work
    """

    def __init__(self, context):
        # print("======== User Define SingleNetwork, context is {}".format(context))
        print("======== User Define SingleNetwork")
        pass

    def build_network(self, context):
        """build net work."""
        super(Network, self).build_network(context)


class Startup(SingleStartup):
    """Start up
    """

    def __init__(self, context):
        # print("======== User Define SingleStartup, context is {}".format(context))
        print("======== User Define SingleStartup")
        pass

    def startup(self, context):
        """build start up.
        """
        super(Startup, self).startup(context)


class Runner(SingleRunner):
    """R
    """

    def __init__(self, context):
        # print("======== User Define SingleRunner, context is {}".format(context))
        print("======== User Define SingleRunner")
        pass

    def run(self, context):
        """build run.
        """
        super(Runner, self).run(context)


class Terminal(TerminalBase):
    """Terminal
    """

    def __init__(self, context):
        # print("======== User Define SingleTerminal, context is {}".format(context))
        print("======== User Define SingleTerminal")
        pass

    def terminal(self, context):
        """build terminal"""
        super(Terminal, self).terminal(context)


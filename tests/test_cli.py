#!/usr/bin/env python
# -*- coding: utf-8 -*-

from click.testing import CliRunner
from grubhub import cli


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'region 1!' in result.output

from invoke import task


@task
def test(c):
    """Run the full Python test suite."""
    c.run(
        "python -m pytest python/tests -q --ignore=python/tests/test_examples.py",
        pty=True,
    )
    c.run(
        (
            "python -c \"import sys, pytest, mlx.core as mx; "
            "mx.set_default_device(mx.cpu); "
            "sys.exit(pytest.main(['python/tests/test_examples.py', '-q']))\""
        ),
        pty=True,
    )
    c.run(
        (
            "python -c \"import sys, pytest, mlx.core as mx; "
            "mx.set_default_device(mx.gpu); "
            "sys.exit(pytest.main(['python/tests/test_examples.py', '-q']))\""
        ),
        pty=True,
    )

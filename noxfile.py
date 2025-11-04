from __future__ import annotations

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the tests.
    """
    pyproject = nox.project.load_toml()
    deps = nox.project.dependency_groups(pyproject, "test")
    session.install("-e.", *deps)
    session.run("pytest", *session.posargs)

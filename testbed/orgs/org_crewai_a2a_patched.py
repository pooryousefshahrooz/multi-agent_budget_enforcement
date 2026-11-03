"""CrewAI organization on CrewAI's own A2A server, with testbed/patches/crewai_a2a_limit.patch applied to CrewAI
(env envs/crewai_a2a_patched). Same application code as org_crewai_a2a; only the CrewAI library differs."""
from orgs.org_crewai_a2a import serve  # noqa: F401

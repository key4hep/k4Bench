"""k4Bench instrumentation for a k4run job.

Appended after the job's own options files (``k4run <job options> <this file>``),
so it changes nothing the job configured. It enables the k4BenchAuditor on every
algorithm and service, and dumps the fully resolved job options. Output paths come
from the environment the k4Bench runner sets:

- ``K4BENCH_COMPONENTS_JSON`` and ``K4BENCH_EVENT_JSON``, read by the auditor;
- ``K4BENCH_JOBOPTIONS``, the resolved options dump.
"""

import os

from Configurables import ApplicationMgr, AuditorSvc, JobOptionsSvc

app = ApplicationMgr()
app.AuditAlgorithms = True
app.AuditServices = True
# First among the external services, so the services created after it, such as
# the geometry, have their initialization audited too.
app.ExtSvc = [AuditorSvc(), *app.ExtSvc]
AuditorSvc().Auditors += ["k4BenchAuditor/k4BenchAuditor"]

if dump := os.environ.get("K4BENCH_JOBOPTIONS"):
    JobOptionsSvc().DUMPFILE = dump

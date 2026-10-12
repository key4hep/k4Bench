These fixtures were read on 2026-10-12 from the Mon devkey-head
x86_64-el9-gcc16-opt LCG view. The two gzipped manifests are unmodified.
`probe.json.gz` is the supplied five-event ILD_FCCee_v02 probe; `loaded.json`
contains versions and buildinfo revisions of its 26 mapped packages.
`HEAD-2026-10-{02,03}.json` are the k4h_packages maps of the stored ILD Zbb
runs on those dates. Their acts change supplies the HEAD compare-link test.

The Geant4 fixture pair uses the documented 11.4.2 → 11.4.3 transition in
lcgcmake commit 5ad19018 / MR !4519. The base version is set explicitly because
its old manifest has rotated away. The head build GITHASH e411040b8 is read
from the actual current Geant4 buildinfo; no base build GITHASH is invented.

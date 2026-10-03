p='src/pool/state.rs'
s=open(p).read()
old = """            WorkerVerdictsWire::Grouped { workers, risks } => bounded(workers, risks),"""
new = """            WorkerVerdictsWire::Grouped { workers, risks } => Self { workers, risks },"""
assert old in s
s = s.replace(old, new, 1)
old2 = """                let (workers, risks) = lines
                    .into_iter()
                    .partition(|line| !line.starts_with("RISK:"));
                bounded(workers, risks)"""
new2 = """                let (workers, risks) = lines
                    .into_iter()
                    .partition(|line| !line.starts_with("RISK:"));
                Self { workers, risks }"""
assert old2 in s
s = s.replace(old2, new2, 1)
# silence the now-unused fn
s = s.replace("fn bounded(workers: Vec<String>, risks: Vec<String>) -> WorkerVerdicts {",
              "#[allow(dead_code)]\nfn bounded(workers: Vec<String>, risks: Vec<String>) -> WorkerVerdicts {", 1)
open(p,'w').write(s)
print("read-path bound temporarily removed")

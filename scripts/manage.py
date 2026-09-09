"""Shared research entry point. Checks never train, access datasets or delete artifacts."""
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={".github","src","configs","scripts","tests","docs","experiments","third_party","workspace"}
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"workspace"/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check():
    cfg=json.loads((ROOT/"project.json").read_text())
    actual={p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in {".git",".venv","build","dist",".pytest_cache"} and not p.name.endswith(".egg-info")}
    if actual!=REQUIRED:raise ValueError(f"Research root directories differ: {actual ^ REQUIRED}")
    if not (ROOT/"src"/cfg["package_name"]).is_dir():raise ValueError("Missing package")
    for name in ["pyproject.toml","framework.lock.json","requirements.txt","README.md","AGENTS.md","CONTRIBUTING.md"]:
        if not (ROOT/name).is_file():raise ValueError("Missing "+name)
    return {"structure":True,"workspace":load("check").check(ROOT/"workspace"),"framework":load("check_framework").check(ROOT)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["check","test","preflight","verify-env"])
    p.add_argument("--machine",default="ibex")
    args=p.parse_args()
    result=check()
    if args.command=="verify-env":
        import importlib, hashlib
        lock=json.loads((ROOT/"framework.lock.json").read_text())
        imported={}
        for item in lock["files"]:
            module=importlib.import_module(item["upstream_path"].removesuffix(".py").replace("/","."))
            path=Path(module.__file__).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:raise ValueError("Installed framework differs: "+str(path))
            imported[module.__name__]=str(path)
        cfg=json.loads((ROOT/"project.json").read_text())
        package=importlib.import_module(cfg["package_name"])
        origin=Path(package.__file__).resolve()
        if not origin.is_relative_to(ROOT/"src"):raise ValueError("Expected this editable application: "+str(origin))
        result["installed"]={"framework":imported,"application":str(origin)}
    if args.command=="test":
        subprocess.run([sys.executable,"-m","pytest","tests/unit"],cwd=ROOT,check=True)
    if args.command=="preflight":
        cfg=json.loads((ROOT/"project.json").read_text())
        deploy=json.loads((ROOT/"configs/deployment"/(args.machine+".json")).read_text())
        registry=json.loads((ROOT/"workspace/registry.json").read_text())
        if deploy["schema"]!="research_deployment_v1" or deploy["project"]!=cfg["project_name"]:raise ValueError("Deployment identity differs")
        if any(deploy[k]!=registry["machines"][args.machine][k] for k in ("code_root","data_root")):raise ValueError("Deployment roots differ")
        result["paths"]=load("paths").resolve(registry,args.machine,cfg["project_name"],cfg["release_id"])
        result.update(data_inspected=False,gpu_requested=False,training_started=False,scope="structure and paths only")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

"""Shared research entry point. Checks never train, access datasets or delete artifacts."""
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={".github","src","configs","scripts","tests","docs","experiments","third_party","workspace"}
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"workspace"/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check():
    spec=importlib.util.spec_from_file_location("internal_layout",ROOT/"scripts/check_layout.py")
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    internal=layout.check(ROOT)
    cfg=json.loads((ROOT/"project.json").read_text())
    actual={p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in {".git",".venv","build","dist",".pytest_cache"} and not p.name.endswith(".egg-info")}
    if actual!=REQUIRED:raise ValueError(f"Research root directories differ: {actual ^ REQUIRED}")
    if not (ROOT/"src"/cfg["package_name"]).is_dir():raise ValueError("Missing package")
    for name in ["pyproject.toml","framework.lock.json","requirements.txt","README.md","AGENTS.md","CONTRIBUTING.md"]:
        if not (ROOT/name).is_file():raise ValueError("Missing "+name)
    return {"structure":True,"internal_layout":internal,"workspace":load("check").check(ROOT/"workspace"),"framework":load("check_framework").check(ROOT)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["check","test","preflight","verify-env"])
    p.add_argument("--machine",default="local")
    p.add_argument("--profile",type=Path,help="Explicit deployment JSON; relative to the project root")
    args=p.parse_args()
    result=check()
    if args.command=="verify-env":
        import importlib, hashlib
        lock=json.loads((ROOT/"framework.lock.json").read_text())
        imported={}
        for item in lock["files"]:
            module=importlib.import_module(item.get("module",item["upstream_path"].removesuffix(".py").replace("/",".")))
            path=Path(module.__file__).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:raise ValueError("Installed framework differs: "+str(path))
            imported[module.__name__]=str(path)
        framework=importlib.import_module("mhd_framework")
        if framework.__api_version__!=lock["api_version"]:raise ValueError("Installed MHD API differs")
        cfg=json.loads((ROOT/"project.json").read_text())
        package=importlib.import_module(cfg["package_name"])
        origin=Path(package.__file__).resolve()
        if not origin.is_relative_to(ROOT/"src"):raise ValueError("Expected this editable application: "+str(origin))
        result["installed"]={"framework":imported,"application":str(origin)}
    if args.command=="test":
        subprocess.run([sys.executable,"-m","pytest","tests/unit"],cwd=ROOT,check=True)
    if args.command=="preflight":
        cfg=json.loads((ROOT/"project.json").read_text())
        if not args.machine.replace("_", "").replace("-", "").isalnum():raise ValueError("Invalid machine name")
        profile=args.profile or Path("configs/deployment")/(args.machine+".json")
        profile=profile if profile.is_absolute() else ROOT/profile
        deploy=json.loads(profile.read_text())
        registry=json.loads((ROOT/"workspace/registry.json").read_text())
        if deploy["schema"]!="research_deployment_v1" or deploy["project"]!=cfg["project_name"]:raise ValueError("Deployment identity differs")
        if args.profile is None and args.machine in registry["machines"] and any(deploy[k]!=registry["machines"][args.machine][k] for k in ("code_root","data_root")):raise ValueError("Deployment roots differ")
        roots=dict(deploy)
        for key in ("code_root","data_root"):
            path=Path(deploy[key]).expanduser()
            roots[key]=str((path if path.is_absolute() else ROOT/path).resolve())
        registry["machines"][args.machine]=roots
        result["paths"]=load("paths").resolve(registry,args.machine,cfg["project_name"],cfg["release_id"])
        result.update(data_inspected=False,gpu_requested=False,training_started=False,scope="structure and paths only")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

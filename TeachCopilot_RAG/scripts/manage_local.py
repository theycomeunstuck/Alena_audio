#!/usr/bin/env python3
"""Manage the isolated local TeachCopilot PostgreSQL database."""
from __future__ import annotations
import argparse, getpass, json, os, subprocess, sys
import psycopg2
from pathlib import Path
from urllib.parse import urlsplit, unquote
from psycopg2.extensions import parse_dsn, make_dsn
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from local_app.store import Store
from local_app.config import Settings
from local_app.auth import hash_password

def dsn_arg(parser,required=True):
    parser.add_argument("--dsn",default=Settings.from_env().database_url if not os.getenv("TEACHCOPILOT_DATABASE_URL") else os.getenv("TEACHCOPILOT_DATABASE_URL"),required=False,help="PostgreSQL DSN (or DATABASE_URL from .local.env)")

def database_name(dsn):
    name=parse_dsn(dsn).get("dbname")
    if not name:
        name=unquote(urlsplit(dsn).path.lstrip("/"))
    if not name: raise ValueError("DSN must include a database name")
    return name

def validate_container_target(dsn,container,runner=subprocess.run):
    """Require the DSN to point at this local container's published Postgres port."""
    params=parse_dsn(dsn)
    host=(params.get("host") or "localhost").strip("[]").lower()
    if host not in {"localhost","127.0.0.1","::1"}:
        raise ValueError("backup and restore require a loopback PostgreSQL DSN")
    if params.get("user") != "teachcopilot":
        raise ValueError("backup and restore require the teachcopilot database role")
    if not params.get("dbname"):
        raise ValueError("DSN must include a database name")
    port=str(params.get("port") or "5432")
    result=runner(["docker","inspect",container],capture_output=True,text=True,check=True)
    inspected=json.loads(result.stdout)
    if len(inspected)!=1:
        raise ValueError("Docker did not return exactly one database container")
    bindings=(inspected[0].get("NetworkSettings",{}).get("Ports",{}).get("5432/tcp") or [])
    for binding in bindings:
        bind_host=binding.get("HostIp") or ""
        if str(binding.get("HostPort"))==port and bind_host in {"127.0.0.1","::1","localhost"}:
            return params["dbname"]
    raise ValueError("DSN host and port do not match the selected container's loopback-published PostgreSQL port")

def target_dsn_for_database(database,configured_dsn=None):
    if not database or not database.strip():
        raise ValueError("target database name is required")
    base=configured_dsn or Settings.from_env().database_url
    return make_dsn(base,dbname=database.strip())

def build_parser():
    p=argparse.ArgumentParser(description="Local TeachCopilot database management")
    sub=p.add_subparsers(dest="command",required=True)
    m=sub.add_parser("migrate"); dsn_arg(m)
    i=sub.add_parser("import"); dsn_arg(i); i.add_argument("--repo-root",type=Path,default=ROOT.parent)
    b=sub.add_parser("bootstrap"); dsn_arg(b); b.add_argument("--email",required=True); b.add_argument("--display-name",required=True); b.add_argument("--password-stdin",action="store_true",help="read password from stdin")
    e=sub.add_parser("export"); dsn_arg(e); e.add_argument("--out",type=Path,required=True)
    bk=sub.add_parser("backup"); dsn_arg(bk); bk.add_argument("--out",type=Path,required=True); bk.add_argument("--container",default=os.getenv("TEACHCOPILOT_DB_CONTAINER","teachcopilot_local_db"))
    rs=sub.add_parser("restore"); target=rs.add_mutually_exclusive_group(required=True); target.add_argument("--target-dsn"); target.add_argument("--target-db",help="existing database name on DATABASE_URL's local PostgreSQL server")
    rs.add_argument("--in",dest="input",type=Path,required=True); rs.add_argument("--container",default=os.getenv("TEACHCOPILOT_DB_CONTAINER","teachcopilot_local_db")); rs.add_argument("--yes",action="store_true",help="confirm restore to the verified empty target database")
    return p

def run(args=None):
    a=build_parser().parse_args(args)
    if a.command=="restore":
        target_dsn=a.target_dsn if a.target_dsn is not None else target_dsn_for_database(a.target_db)
        dbname=validate_container_target(target_dsn,a.container)
        # Guard against accidental overwrite: only restore into an empty database.
        s=Store(target_dsn)
        try:
            with s._connect() as conn,conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM pg_stat_user_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')")
                if cur.fetchone()["n"]: raise SystemExit("Refusing restore: target database contains user tables")
        except psycopg2.Error as exc:
            raise SystemExit("Could not verify the local restore target. Confirm that the named database already exists and local database settings are valid.") from None
        if not a.yes: raise SystemExit("Target is empty. Rerun with --yes to restore.")
        with a.input.open("rb") as backup:
            subprocess.run(["docker","exec","-i",a.container,"pg_restore","--exit-on-error","--no-owner","--no-privileges","--username","teachcopilot","--dbname",dbname],stdin=backup,check=True)
        print("Restore completed."); return 0
    store=Store(a.dsn)
    if a.command=="migrate":
        print(json.dumps({"migrations_applied":store.migrate()})); return 0
    if a.command=="import":
        store.migrate(); print(json.dumps(store.import_repository(a.repo_root),ensure_ascii=False)); return 0
    if a.command=="bootstrap":
        store.migrate()
        password=sys.stdin.read().strip() if a.password_stdin else getpass.getpass("Teacher password: ")
        try: hashed=hash_password(password)
        except ValueError as exc: raise SystemExit(str(exc)) from exc
        account=store.create_account(a.email,a.display_name,hashed,"teacher")
        for card in store.list_cards(): store.assign_learner(account["id"],card["learner_id"],"teacher")
        print(json.dumps({k:str(v) for k,v in account.items() if k!="password_hash"},ensure_ascii=False)); return 0
    if a.command=="export":
        a.out.mkdir(parents=True,exist_ok=True)
        out=a.out/"learners"; out.mkdir(exist_ok=True)
        for row in store.list_cards():
            card=store.get_card(row["learner_id"])
            (out/f"{row['learner_id']}.json").write_text(json.dumps(card,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (a.out/"catalog.json").write_text(json.dumps(store.catalog(),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(f"Exported {len(store.list_cards())} learner cards to {a.out}"); return 0
    if a.command=="backup":
        dbname=validate_container_target(a.dsn,a.container)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("wb") as backup:
            subprocess.run(["docker","exec",a.container,"pg_dump","--format=custom","--no-owner","--no-privileges","--username","teachcopilot","--dbname",dbname],stdout=backup,check=True)
        print(f"Backup written to {a.out}"); return 0
    return 2

if __name__=="__main__":
    try: raise SystemExit(run())
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"manage_local: {exc}",file=sys.stderr); raise SystemExit(1)


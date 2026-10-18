"""Verify paired evidence and a local-model proposal in a disposable database."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, default=ROOT / '.local' / 'rich-demo-credentials.json')
    parser.add_argument('--output', type=Path, default=ROOT / '.local' / 'verification' / 'live-workflow.json')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json, subprocess, uuid
    from datetime import date
    from dataclasses import replace
    from urllib.parse import urlsplit
    import psycopg2, requests
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn, parse_dsn
    from fastapi.testclient import TestClient
    from local_app.config import Settings
    from local_app.api import create_app
    from local_app.store import Store
    settings=Settings.from_env()
    assert parse_dsn(settings.database_url).get('host') in {'127.0.0.1','localhost','::1'},'Live verification requires a loopback database'
    name='teachcopilot_enriched_acceptance_'+uuid.uuid4().hex[:10]
    assert name.startswith('teachcopilot_enriched_acceptance_')
    admin=psycopg2.connect(make_dsn(settings.database_url,dbname='postgres'));admin.autocommit=True
    private=json.loads(args.credentials.read_text(encoding='utf-8'))
    out={}
    with admin.cursor() as c:c.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    try:
     for command_args in (['backup','--out','.local/backups/enriched.dump'],['restore','--target-db',name,'--in','.local/backups/enriched.dump','--yes']):
      p=subprocess.run([sys.executable,'scripts/manage_local.py',*command_args],capture_output=True,text=True,encoding='utf-8')
      assert p.returncode==0,'Disposable backup/restore failed'
     target=replace(settings,database_url=make_dsn(settings.database_url,dbname=name))
     original=requests.sessions.Session.request
     def loopback(self,method,url,*args,**kwargs):
      assert urlsplit(str(url)).hostname in {'127.0.0.1','localhost','::1'},'Nonlocal HTTP forbidden'
      return original(self,method,url,*args,**kwargs)
     requests.sessions.Session.request=loopback
     db=Store(target.database_url);client=TestClient(create_app(settings=target,store=db))
     r=client.post('/auth/login',json=private['teacher']);r.raise_for_status()
     h={'Authorization':'Bearer '+r.json()['access_token']}
     learner='demo-rita-04';topic='math.g4.quantities.area_units'
     revision=db.get_revision(learner);initial=revision
     for pair in ('acceptance-area-a','acceptance-area-b'):
      for support,solved in (('independent',False),('visual',True)):
       payload={'expected_revision':revision,'topic_id':topic,'task_id':pair+'-'+support,'assessment_pair_id':pair,'difficulty':'medium','support_level':support,'solved':solved,'notes':'\u041f\u0430\u0440\u0430 \u0441\u043e\u043f\u043e\u0441\u0442\u0430\u0432\u0438\u043c\u044b\u0445 \u0437\u0430\u0434\u0430\u043d\u0438\u0439 \u043f\u043e \u0435\u0434\u0438\u043d\u0438\u0446\u0430\u043c \u043f\u043b\u043e\u0449\u0430\u0434\u0438.','observed_on':date.today().isoformat()}
       r=client.post('/learners/'+learner+'/observations',headers=h,json=payload)
       r.raise_for_status();revision=db.get_revision(learner)
     stale=client.post('/learners/'+learner+'/observations',headers=h,json={'expected_revision':initial,'topic_id':topic,'task_id':'stale-example','support_level':'independent','solved':True})
     assert stale.status_code==409
     d=client.get('/learners/'+learner+'/dashboard',headers=h);d.raise_for_status()
     row=next(x for x in d.json()['zpd']['topics'] if x['topic_id']==topic)
     assert row['confirmed_zone']=='zpd' and row['evidence_status']=='confirmed'
     out['after_assisted_pairs']={'confirmed_zone':row['confirmed_zone'],'revision':revision}
     for pair in ('acceptance-area-a','acceptance-area-b'):
      r=client.post('/learners/'+learner+'/observations',headers=h,json={'expected_revision':revision,'topic_id':topic,'task_id':pair+'-transfer','assessment_pair_id':pair,'difficulty':'medium','support_level':'independent','solved':True,'notes':'\u041d\u043e\u0432\u044b\u0439 \u0441\u043e\u043f\u043e\u0441\u0442\u0430\u0432\u0438\u043c\u044b\u0439 \u043f\u0440\u0438\u043c\u0435\u0440 \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d \u0431\u0435\u0437 \u043f\u043e\u0434\u0441\u043a\u0430\u0437\u043a\u0438.','observed_on':date.today().isoformat()})
      r.raise_for_status();revision=db.get_revision(learner)
     d=client.get('/learners/'+learner+'/dashboard',headers=h);d.raise_for_status()
     row=next(x for x in d.json()['zpd']['topics'] if x['topic_id']==topic)
     assert row['confirmed_zone']=='mastered'
     out['after_fading']={'confirmed_zone':row['confirmed_zone'],'revision':revision}
     notes='\u0420\u0438\u0442\u0430 \u043d\u0430 \u0437\u0430\u043d\u044f\u0442\u0438\u0438 \u0441\u0435\u0433\u043e\u0434\u043d\u044f \u0440\u0430\u0437\u0431\u0438\u0440\u0430\u043b\u0430 \u0435\u0434\u0438\u043d\u0438\u0446\u044b \u043f\u043b\u043e\u0449\u0430\u0434\u0438. \u0421\u0430\u043c\u043e\u0441\u0442\u043e\u044f\u0442\u0435\u043b\u044c\u043d\u043e \u0432\u044b\u043f\u043e\u043b\u043d\u0438\u043b\u0430 \u0434\u0432\u0430 \u0441\u043e\u043f\u043e\u0441\u0442\u0430\u0432\u0438\u043c\u044b\u0445 \u043f\u0440\u0438\u043c\u0435\u0440\u0430 \u043f\u0435\u0440\u0435\u0432\u043e\u0434\u0430 \u043a\u0432\u0430\u0434\u0440\u0430\u0442\u043d\u044b\u0445 \u0441\u0430\u043d\u0442\u0438\u043c\u0435\u0442\u0440\u043e\u0432 \u0438 \u043e\u0431\u044a\u044f\u0441\u043d\u0438\u043b\u0430, \u043f\u043e\u0447\u0435\u043c\u0443 \u043d\u0443\u0436\u043d\u043e \u0443\u0447\u0438\u0442\u044b\u0432\u0430\u0442\u044c \u0434\u0432\u0435 \u0441\u0442\u043e\u0440\u043e\u043d\u044b \u043a\u0432\u0430\u0434\u0440\u0430\u0442\u0430. \u041d\u0430 \u0432\u043e\u043f\u0440\u043e\u0441\u044b \u043e\u0442\u0432\u0435\u0447\u0430\u043b\u0430 \u0431\u0435\u0437 \u043f\u043e\u0434\u0441\u043a\u0430\u0437\u043e\u043a. \u0417\u0430\u043d\u0438\u043c\u0430\u043b\u0430\u0441\u044c 20 \u043c\u0438\u043d\u0443\u0442, \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u0438\u0435 \u0441\u043f\u043e\u043a\u043e\u0439\u043d\u043e\u0435, \u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u0438\u043b\u0430 \u0440\u0438\u0441\u0443\u043d\u043e\u043a-\u0441\u0445\u0435\u043c\u0443.'
     before=db.get_revision(learner)
     proposal=client.post('/learner/update',headers=h,json={'learner_ids':[learner],'notes':notes})
     proposal.raise_for_status();pid=proposal.json()['proposal_id']
     assert pid and db.get_revision(learner)==before
     preview=client.get('/lessons/proposals/'+pid,headers=h);preview.raise_for_status()
     r=client.post('/lessons/proposals/'+pid+'/confirm',headers=h);r.raise_for_status()
     after=db.get_revision(learner);assert after==before+1
     r=client.post('/lessons/proposals/'+pid+'/confirm',headers=h);r.raise_for_status()
     assert db.get_revision(learner)==after
     out['local_model_proposal']={'preview':200,'confirmed':True,'idempotent':True,'revision_increment':1,'model':target.chat_model}
     client.post('/auth/logout',headers=h).raise_for_status()
     out['passed']=True;out['database']='disposable restored copy';out['http_egress']='loopback only'
     args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
     print('Local acceptance passed: paired evidence, revision conflict, Gemma proposal, idempotent confirmation')
    finally:
     with admin.cursor() as c:
      c.execute('SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()',(name,))
      c.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))
     admin.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

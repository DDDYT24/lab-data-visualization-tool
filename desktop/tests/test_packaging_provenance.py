"""Build and license evidence must fail on changed inputs instead of being inferred."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGING = Path(__file__).resolve().parents[1]/'packaging'
sys.path.insert(0, str(PACKAGING))
import provenance
spec = importlib.util.spec_from_file_location('desktop_license_collector', PACKAGING/'licenses.py')
licenses = importlib.util.module_from_spec(spec); spec.loader.exec_module(licenses)


class PackagingProvenanceTests(unittest.TestCase):
    def supplement_fixture(self, root):
        package=root/'package';supplements=root/'supplements'
        metadata={'name':'example','version':'1.0.0','license':'MIT'}
        manifest=self.put(package,'package.json',json.dumps(metadata))
        implementation=self.put(package,'index.js','module.exports = 1;')
        text=self.put(supplements,'upstream.txt','Authentic upstream permission and copyright text')
        item={'name':'example','version':'1.0.0','packageMetadataSha256':provenance.sha(manifest),
              'implementationSha256':{'index.js':provenance.sha(implementation)},'noticeFiles':['upstream.txt'],
              'noticeSourcePackageVersion':'2.0.0','releaseContainedFullNotice':False}
        source={'files':{'upstream.txt':{'sha256':provenance.sha(text)}},'webPackages':[item]}
        return package,metadata,supplements,source

    def test_web_supplement_keeps_its_later_upstream_origin_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            package,metadata,supplements,source=self.supplement_fixture(Path(folder))
            paths,origin=licenses.web_supplement(package,metadata,supplements,source)
            self.assertEqual(paths,[supplements/'upstream.txt'])
            self.assertEqual(origin['noticeSourcePackageVersion'],'2.0.0')
            self.assertFalse(origin['releaseContainedFullNotice'])
            self.assertEqual(licenses.web_supplement(package,{**metadata,'version':'3.0.0'},supplements,source),([],None))

    def test_web_supplement_refuses_changed_inputs_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package,metadata,supplements,source=self.supplement_fixture(root)
            for path,message in [(package/'package.json','metadata differs'),(package/'index.js','implementation differs'),(supplements/'upstream.txt','notice hash differs')]:
                original=path.read_bytes();path.write_bytes(b'changed')
                with self.subTest(path=path),self.assertRaisesRegex(ValueError,message):
                    licenses.web_supplement(package,metadata,supplements,source)
                path.write_bytes(original)
            outside=self.put(root,'outside.txt','preserve')
            source['webPackages'][0]['noticeFiles']=['../outside.txt']
            source['files']['../outside.txt']={'sha256':provenance.sha(outside)}
            with self.assertRaisesRegex(ValueError,'notice hash differs'):
                licenses.web_supplement(package,metadata,supplements,source)

    def test_unicode_web_metadata_is_read_with_utf8_disabled_in_a_real_child(self):
        import os,subprocess
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();bundle=root/'bundle';bundle.mkdir()
            self.put(root,'desktop/packaging/license-supplements/sources.json','{"files":{}}')
            self.put(root,'V2.0/web/package-lock.json',json.dumps({'packages':{'node_modules/example':{'version':'1'}}}))
            self.put(root,'V2.0/web/node_modules/example/package.json',json.dumps({'name':'example','version':'1','license':'MIT','author':'中文研究室 – α'},ensure_ascii=False))
            self.put(root,'V2.0/web/node_modules/example/LICENSE','Copyright Example. Permission is hereby granted.')
            code='import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import licenses;result=licenses.web_inventory(Path(sys.argv[2]),Path(sys.argv[3]));assert len(result["packages"])==1 and not result["reviewIssues"]'
            result=subprocess.run([sys.executable,'-B','-X','utf8=0','-c',code,str(PACKAGING),str(root),str(bundle)],
                env={**os.environ,'PYTHONUTF8':'0','PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8',timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def build_helper(self):
        path=PACKAGING.parent/'scripts/build_verified.py'
        spec=importlib.util.spec_from_file_location('verified_builder_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_build_output_cannot_write_into_candidates_or_unrelated_directories(self):
        helper=self.build_helper()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();boundary=root/'outputs';boundary.mkdir()
            for requested in [root, boundary, root.parent/'outside']:
                with self.subTest(path=requested),self.assertRaises(ValueError):helper.build_directory(root,requested)
            for marker in ['desktop-package.json','desktop-runtime.json','candidate.json']:
                candidate=boundary/marker.removesuffix('.json');candidate.mkdir();self.put(candidate,marker,'preserve')
                with self.subTest(marker=marker),self.assertRaises(ValueError):helper.build_directory(root,candidate/'new-receipts')
                self.assertFalse((candidate/'new-receipts').exists())
            directory=boundary/'notes';self.put(directory,'important.txt','preserve')
            with self.assertRaises(ValueError):helper.build_directory(root,directory)
            self.assertEqual((directory/'important.txt').read_text(),'preserve')
            fresh=boundary/'new';self.assertEqual(helper.build_directory(root,fresh),fresh)
            self.put(fresh,'host.json','first component');self.put(fresh,'host.log','log')
            self.assertEqual(helper.build_directory(root,fresh),fresh)

    @unittest.skipUnless(__import__('os').name=='nt','Windows junction boundary')
    def test_build_output_rejects_junctions_before_resolving_them(self):
        helper=self.build_helper()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();boundary=root/'outputs';boundary.mkdir();actual=boundary/'actual';actual.mkdir()
            link=boundary/'linked'
            __import__('subprocess').run(['cmd','/c','mklink','/J',str(link),str(actual)],capture_output=True,check=True)
            try:
                with self.assertRaisesRegex(ValueError,'Linked'):helper.build_directory(root,link/'receipts')
                self.assertEqual(list(actual.iterdir()),[])
            finally:link.rmdir()

    def put(self, root, name, content):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding='utf-8'); return path

    def fixture(self, root):
        for name in ['desktop/host/src/main.rs','desktop/host/boot/index.html','desktop/host/Cargo.toml','desktop/host/Cargo.lock','desktop/host/build.rs','desktop/host/tauri.conf.json','desktop/runtime/adapter.js','V2.0/assets/labviz-logo.ico','desktop/scripts/build_verified.py','V2.0/web/src/app.tsx','V2.0/web/package.json','V2.0/web/node_modules/.package-lock.json','desktop/prototypes/tauri/target/release/LabViz.exe','V2.0/web/.next-v23-package/standalone/server.js','V2.0/web/.next-v23-package/static/chunk.js','V2.0/web/.next-v23-package/BUILD_ID']:
            self.put(root, name, 'fixture')
        self.put(root,'desktop/host/permissions/acceptance-ready.toml',
                 (PACKAGING.parent/'host/permissions/acceptance-ready.toml').read_text(encoding='utf-8'))
        host_out=root/'desktop/prototypes/tauri/target/release/build/labviz-desktop-fixture/out'
        self.put(host_out,'acl-manifests.json','{}');self.put(host_out,'capabilities.json','{}')
        directory = root/'receipts'; directory.mkdir()
        for kind in ['host','web']:
            content=json.dumps({'reason':'build-script-executed','package_id':'path+file:///fixture#labviz-desktop@1','out_dir':str(host_out)}) if kind=='host' else 'actual build log'
            log = self.put(directory, kind+'.log', content)
            snapshot = provenance.source(root, kind)
            value = {'format':'labviz-component-build-v1','component':kind,'exitCode':0,'buildToolsUnchanged':True,
                     'sourceBefore':snapshot,'sourceAfter':snapshot,'outputs':provenance.outputs(root,kind,host_out if kind=='host' else None),
                     'builderSha256':provenance.sha(root/'desktop/scripts/build_verified.py'),
                     'provenanceToolSha256':provenance.sha(PACKAGING/'provenance.py'),'logSha256':provenance.sha(log)}
            if kind=='host':value['hostBuildScriptOutDir']=str(host_out)
            self.put(directory, kind+'.json', json.dumps(value))
        return directory

    def test_receipts_bind_sources_outputs_and_the_actual_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); directory = self.fixture(root)
            self.assertEqual(set(provenance.validate(root,directory)),{'host','web'})
            for name in ['desktop/host/src/main.rs','desktop/runtime/adapter.js','V2.0/web/src/app.tsx','desktop/prototypes/tauri/target/release/LabViz.exe','V2.0/web/.next-v23-package/static/chunk.js','receipts/host.log']:
                path = root/name; original = path.read_bytes(); path.write_bytes(b'changed input')
                with self.subTest(file=name), self.assertRaises(ValueError): provenance.validate(root,directory)
                path.write_bytes(original)
            self.assertEqual(set(provenance.validate(root,directory)),{'host','web'})

    def test_permission_inputs_and_resolved_acl_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();directory=self.fixture(root)
            for name in ['desktop/host/permissions/acceptance-ready.toml',
                         'desktop/prototypes/tauri/target/release/build/labviz-desktop-fixture/out/acl-manifests.json',
                         'desktop/prototypes/tauri/target/release/build/labviz-desktop-fixture/out/capabilities.json']:
                path=root/name;original=path.read_bytes();path.write_bytes(original+b'\n# changed')
                with self.subTest(path=name),self.assertRaises(ValueError):provenance.validate(root,directory)
                path.write_bytes(original)

    def test_extra_and_residual_permissions_are_rejected(self):
        for name in ['extra.toml','autogenerated/acceptance_editor_ready.toml']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();self.fixture(root);self.put(root,'desktop/host/permissions/'+name,'# unexpected')
                with self.assertRaisesRegex(ValueError,'Unexpected'):provenance.source(root,'host')

    def test_duplicate_and_broader_permission_commands_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();self.fixture(root);path=root/'desktop/host/permissions/acceptance-ready.toml';original=path.read_text(encoding='utf-8')
            path.write_text(original.replace('deny-acceptance-editor-ready','allow-acceptance-editor-ready'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Duplicate'):provenance.source(root,'host')
            path.write_text(original.replace('commands.allow = ["acceptance_editor_ready"]','commands.allow = ["acceptance_editor_ready", "other_command"]'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'commands or scope'):provenance.source(root,'host')

    def test_cargo_output_directory_is_selected_from_the_actual_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();directory=self.fixture(root);log=directory/'host.log';original=log.read_text()
            log.write_text('not a cargo event')
            with self.assertRaisesRegex(ValueError,'missing or ambiguous'):provenance.host_build_output(root,log)
            item=json.loads(original);item['out_dir']=str(root/'outside/out');log.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError,'escaped'):provenance.host_build_output(root,log)
            item['out_dir']=str(root/'desktop/prototypes/tauri/target/release/build/labviz-desktop-second/out')
            log.write_text(original+'\n'+json.dumps(item))
            with self.assertRaisesRegex(ValueError,'missing or ambiguous'):provenance.host_build_output(root,log)

    def test_staged_host_acl_copies_and_effective_scope_are_verified(self):
        spec=importlib.util.spec_from_file_location('host_evidence_builder',PACKAGING/'build_candidate.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();bundle=root/'bundle';proof=bundle/'provenance'
            out=root/'desktop/prototypes/tauri/target/release/build/labviz-desktop-fixture/out'
            acl={'__app-acl__':{'default_permission':None,'permission_sets':{},'permissions':{
                'allow-acceptance-editor-ready':{'commands':{'allow':['acceptance_editor_ready'],'deny':[]}},
                'deny-acceptance-editor-ready':{'commands':{'allow':[],'deny':['acceptance_editor_ready']}}}}}
            caps={'acceptance-ready':{'identifier':'acceptance-ready','local':False,'windows':['main'],
                'remote':{'urls':['http://127.0.0.1:*']},'permissions':['allow-acceptance-editor-ready']}}
            receipt={'hostBuildScriptOutDir':str(out),'outputs':{'files':{}},'sourceAfter':{'files':{}}}
            for name,value in [('acl-manifests.json',acl),('capabilities.json',caps)]:
                p=self.put(out,name,json.dumps(value));self.put(proof,'host-'+name,p.read_text())
                receipt['outputs']['files'][p.relative_to(root).as_posix()]=provenance.sha(p)
            permission=self.put(proof,'host-permissions.toml','fixed input')
            receipt['sourceAfter']['files']['desktop/host/permissions/acceptance-ready.toml']=provenance.sha(permission)
            with patch.object(builder,'ROOT',root):
                builder.validate_host_evidence(bundle,receipt)
                for p in [permission,proof/'host-acl-manifests.json',proof/'host-capabilities.json']:
                    original=p.read_bytes();p.write_bytes(b'changed')
                    with self.subTest(path=p),self.assertRaisesRegex(ValueError,'differs'):builder.validate_host_evidence(bundle,receipt)
                    p.write_bytes(original)
                caps['acceptance-ready']['webviews']=['startup'];p=self.put(proof,'host-capabilities.json',json.dumps(caps))
                receipt['outputs']['files'][(out/'capabilities.json').relative_to(root).as_posix()]=provenance.sha(p)
                with self.assertRaisesRegex(ValueError,'scope differs'):builder.validate_host_evidence(bundle,receipt)
    def test_failed_or_unstable_build_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); directory = self.fixture(root); path = directory/'host.json'; good=json.loads(path.read_text())
            for field,value in [('exitCode',1),('buildToolsUnchanged',False),('sourceBefore',{})]:
                path.write_text(json.dumps({**good,field:value}))
                with self.subTest(field=field), self.assertRaises(ValueError): provenance.validate(root,directory)

    def test_licence_spellings_and_record_notices_outside_dist_info_are_collected(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle=Path(folder).resolve();site=bundle/'app/runtime/python/Lib/site-packages'
            self.put(site,'openpyxl-1.dist-info/METADATA','Name: openpyxl\nVersion: 1\nLicense: MIT\n')
            self.put(site,'openpyxl-1.dist-info/LICENCE.rst','authentic wheel license')
            self.put(site,'openpyxl-1.dist-info/RECORD','openpyxl/notices/NOTICE.txt,,\n')
            self.put(site,'openpyxl/notices/NOTICE.txt','additional notice')
            rows=licenses.python_inventory(bundle)
            self.assertEqual(len(rows[0]['licenseFiles']),2)
            for name in ['LICENSE','LICENCE','LICENCE.rst','COPYING-MIT','NOTICE.txt']:
                self.assertTrue(licenses.notice(Path(name)))
            self.assertFalse(licenses.notice(Path('unlicensed-source.py')))

    def test_missing_wheel_notice_and_escaping_record_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle=Path(folder).resolve();site=bundle/'app/runtime/python/Lib/site-packages'
            self.put(site,'example.dist-info/METADATA','Name: example\nVersion: 1\n')
            with self.assertRaisesRegex(ValueError,'no collected notice'):licenses.python_inventory(bundle)
            self.put(site,'example.dist-info/RECORD','../../../../../../LICENSE,,\n')
            with self.assertRaisesRegex(ValueError,'escaped'):licenses.python_inventory(bundle)

    def test_web_readme_and_missing_notice_are_both_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();bundle=root/'bundle';bundle.mkdir()
            self.put(root,'desktop/packaging/license-supplements/sources.json','{"files":{}}')
            self.put(root,'V2.0/web/package-lock.json',json.dumps({'packages':{'node_modules/a':{'version':'1'},'node_modules/b':{'version':'1'},'node_modules/missing':{'version':'1','optional':True}}}))
            self.put(root,'V2.0/web/node_modules/a/package.json','{"name":"a","version":"1","license":"MIT"}')
            self.put(root,'V2.0/web/node_modules/a/README.md','Copyright Example. Permission is hereby granted to use this test notice.')
            self.put(root,'V2.0/web/node_modules/b/package.json','{"name":"b","version":"1","license":"MIT"}')
            result=licenses.web_inventory(root,bundle)
            self.assertEqual([x['status'] for x in result['packages']],['readme-notice','declared-license-only','not-installed'])
            self.assertEqual(len(result['reviewIssues']),1);self.assertFalse(result['publicReleaseAccepted'])

    def test_frozen_candidate_refuses_before_writing_notices(self):
        spec=importlib.util.spec_from_file_location('desktop_freezer',PACKAGING/'build_candidate.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();marker=self.put(root,'desktop-package.json','frozen bytes')
            with self.assertRaisesRegex(ValueError,'frozen'):builder.freeze(root,root/'output',root/'cargo')
            self.assertEqual(marker.read_text(),'frozen bytes');self.assertFalse((root/'licenses').exists())

    def test_rust_supplement_requires_exact_version_commit_and_text_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();cargo=root/'cargo';bundle=root/'bundle'
            self.put(root,'desktop/host/Cargo.lock','[[package]]\nname="example"\nversion="1.0.0"\nsource="registry+fixture"\nchecksum="fixture"\n')
            crate=cargo/'registry/src/cache/example-1.0.0'
            self.put(crate,'Cargo.toml','[package]\nname="example"\nversion="1.0.0"\nlicense="MIT"\n')
            self.put(crate,'.cargo_vcs_info.json',json.dumps({'git':{'sha1':'a'*40}}))
            text=self.put(root,'desktop/packaging/license-supplements/MIT.txt','authentic pinned upstream text')
            data={'files':{'MIT.txt':{'sha256':provenance.sha(text),'url':'https://example.invalid/pinned'}},'packages':[{'name':'example','version':'1.0.0','vcsCommit':'a'*40,'file':'MIT.txt'}]}
            info=self.put(root,'desktop/packaging/license-supplements/sources.json',json.dumps(data))
            with patch.object(licenses.subprocess,'check_output',side_effect=['example v1.0.0\n','']):
                rows=licenses.rust_inventory(root,bundle,cargo)
            self.assertEqual(rows[0]['role'],'build-only');self.assertEqual(rows[0]['status'],'version-pinned-upstream')
            for field,value in [('vcsCommit','b'*40),('version','2.0.0')]:
                changed=json.loads(json.dumps(data));changed['packages'][0][field]=value;info.write_text(json.dumps(changed))
                with self.subTest(field=field),patch.object(licenses.subprocess,'check_output',side_effect=['example v1.0.0\n','']),self.assertRaisesRegex(ValueError,'Unresolved Rust notice'):
                    licenses.rust_inventory(root,bundle,cargo)
            info.write_text(json.dumps(data));text.write_text('changed license bytes')
            with patch.object(licenses.subprocess,'check_output',side_effect=['example v1.0.0\n','']),self.assertRaisesRegex(ValueError,'hash differs'):
                licenses.rust_inventory(root,bundle,cargo)

import zipfile

from lake_pinn.source_package import export_source_package, iter_source_files


def test_source_package_filters_results_checkpoints_and_caches(tmp_path):
    root = tmp_path
    (root / 'lake_pinn').mkdir()
    (root / 'lake_pinn' / '__init__.py').write_text('', encoding='utf-8')
    (root / 'lake_pinn' / '__pycache__').mkdir()
    (root / 'lake_pinn' / '__pycache__' / 'mod.pyc').write_bytes(b'cache')
    (root / 'tests').mkdir()
    (root / 'tests' / 'test_smoke.py').write_text('def test_smoke(): pass\n', encoding='utf-8')
    (root / 'experiments' / 'splits').mkdir(parents=True)
    (root / 'experiments' / 'splits' / 'split.json').write_text('{}\n', encoding='utf-8')
    (root / 'results').mkdir()
    (root / 'results' / 'history.csv').write_text('epoch,loss\n', encoding='utf-8')
    (root / '.pytest_cache').mkdir()
    (root / '.pytest_cache' / 'README.md').write_text('cache\n', encoding='utf-8')
    (root / 'checkpoint.pt').write_bytes(b'checkpoint')
    (root / '.source_packageignore').write_text(
        'results/\n.pytest_cache/\n__pycache__/\n*.pt\n',
        encoding='utf-8',
    )

    files = {path.as_posix() for path in iter_source_files(root)}

    assert 'lake_pinn/__init__.py' in files
    assert 'tests/test_smoke.py' in files
    assert 'experiments/splits/split.json' in files
    assert not any(path.startswith('results/') for path in files)
    assert not any('.pytest_cache' in path for path in files)
    assert not any('__pycache__' in path for path in files)
    assert 'checkpoint.pt' not in files


def test_export_source_package_writes_only_clean_source_files(tmp_path):
    root = tmp_path
    (root / 'lake_pinn').mkdir()
    (root / 'lake_pinn' / '__init__.py').write_text('', encoding='utf-8')
    (root / 'results').mkdir()
    (root / 'results' / 'metric.csv').write_text('rmse\n', encoding='utf-8')
    output_path = tmp_path / 'dist' / 'package.zip'

    export_source_package(root, output_path)

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())

    assert names == {'lake_pinn/__init__.py'}

import pytest


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    monkeypatch.setenv('PYFR_HIP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.delenv('PYFR_HIP_DISABLE_CACHE', raising=False)

    try:
        from pyfr.backends.hip.compiler import HIPCompiler
        from pyfr.backends.hip.driver import HIP

        compiler = HIPCompiler(HIP())
    except OSError:
        pytest.skip('HIP runtime or HIPRTC not available')

    # Count the compilations which miss the cache
    compiler.ncompile = 0
    compile = compiler.hiprtc.compile

    def counted(*args, **kwargs):
        compiler.ncompile += 1
        return compile(*args, **kwargs)

    compiler.hiprtc.compile = counted

    return compiler


def test_header_edit_invalidates_cache(compiler, tmp_path):
    idir = tmp_path / 'include'
    (idir / 'sub').mkdir(parents=True)
    hdr = idir / 'sub' / 'val.hpp'

    src = ('#include "sub/val.hpp"\n'
           'extern "C" __global__ void f(int *x) { *x = VAL; }\n')
    flags = ['--gpu-architecture=gfx942', f'-I{idir}']

    # A first build compiles and a second is served from the cache
    hdr.write_text('#define VAL 1\n')
    code1 = compiler.build('kernel', src, flags)
    assert compiler.build('kernel', src, flags) == code1
    assert compiler.ncompile == 1

    # Editing the header must force a rebuild with new code
    hdr.write_text('#define VAL 2\n')
    code2 = compiler.build('kernel', src, flags)
    assert compiler.ncompile == 2
    assert code2 != code1

    # The separate '-I dir' form is keyed on the headers too
    hdr.write_text('#define VAL 3\n')
    code3 = compiler.build('kernel', src, flags[:1] + ['-I', str(idir)])
    assert compiler.ncompile == 3
    assert code3 not in (code1, code2)

    # Reverting the header hits the cache entry of the original build
    hdr.write_text('#define VAL 1\n')
    assert compiler.build('kernel', src, flags) == code1
    assert compiler.ncompile == 3

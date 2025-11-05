import pathlib
import shutil
import sys

# test_old = True
test_old = False

TESTING_DIR = pathlib.Path(__file__).resolve().parent
OPTIONS_DIR = TESTING_DIR.joinpath('options')

if test_old:
    BADASS_DIR = pathlib.Path('/Users/sara/Dropbox/research/bgc/badass_repo')
else:
    BADASS_DIR = TESTING_DIR.parent

NOTEBOOKS_DIR = BADASS_DIR.joinpath('example_notebooks')
EX_SPEC_DIR = BADASS_DIR.joinpath('examples', 'example_spectra')

sys.path.insert(0, str(BADASS_DIR))
import badass


def test_single():
    options_file = OPTIONS_DIR.joinpath('sdss_single.py')

    test_file = EX_SPEC_DIR.joinpath('0-test', 'spec-1087-52930-0084.fits')
    # test_file = EX_SPEC_DIR.joinpath('1-test', 'spec-7748-58396-0782.fits')
    # test_file = EX_SPEC_DIR.joinpath('2-test', 'spec-2756-54508-0579.fits')

    if test_old:
        output_dir = test_file.parent.joinpath('sdss_test_oldrepo_ml')
    else:
        output_dir = test_file.parent.joinpath('sdss_test')

    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    if test_old:
        badass.run_BADASS(test_file, run_dir=output_dir, options_file=options_file, sdss_spec=True)
    else:
        badass.run_BADASS(test_file, options_file=options_file)


def test_line():
    options_file = OPTIONS_DIR.joinpath('line_test.py')
    # test_file = EX_SPEC_DIR.joinpath('0-test', 'spec-1087-52930-0084.fits')
    test_file = EX_SPEC_DIR.joinpath('1-test', 'spec-7748-58396-0782.fits')

    output_dir = test_file.parent.joinpath('line_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    badass.run_BADASS(test_file, options_file=options_file)


def test_config():
    options_file = OPTIONS_DIR.joinpath('config_test.py')
    test_file = EX_SPEC_DIR.joinpath('0-test', 'spec-1087-52930-0084.fits')

    output_dir = test_file.parent.joinpath('config_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    badass.run_BADASS(test_file, options_file=options_file)


def test_muse_single():
    options_file = OPTIONS_DIR.joinpath('muse_single.py')
    test_file = EX_SPEC_DIR.joinpath('MUSE', 'NGC1068_subcube.fits')
    output_dir = test_file.parent.joinpath('muse_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    badass.run_BADASS(test_file, options_file=options_file)


def test_muse_multi():
    options_file = OPTIONS_DIR.joinpath('muse_multi.py')
    test_file = EX_SPEC_DIR.joinpath('MUSE', 'NGC1068_subcube.fits')
    output_dir = test_file.parent.joinpath('muse_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    # badass.target_check(test_file, options_file=options_file)
    badass.run_BADASS(test_file, options_file=options_file, nprocesses=2)



def test_nirspec_single():
    options_file = OPTIONS_DIR.joinpath('nirspec_single.py')
    test_file = EX_SPEC_DIR.joinpath('JWST_NIRSpec', 'NGC4051_nirspec_290_s3d.fits')
    output_dir = test_file.parent.joinpath('nirspec_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    badass.run_BADASS(test_file, options_file=options_file)


def test_nirspec_aperture():
    options_file = OPTIONS_DIR.joinpath('nirspec_aperture.py')
    test_file = EX_SPEC_DIR.joinpath('JWST_NIRSpec', 'NGC4051_nirspec_290_s3d.fits')
    output_dir = test_file.parent.joinpath('ns_ap_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    badass.target_check(test_file, options_file=options_file)
    # badass.run_BADASS(test_file, options_file=options_file)


def test_miri_single():
    options_file = OPTIONS_DIR.joinpath('miri_single.py')
    test_file = EX_SPEC_DIR.joinpath('JWST_MIRI', 'NGC4051_miri_ch1-short_s3d.fits')
    output_dir = test_file.parent.joinpath('miri_test')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))

    badass.run_BADASS(test_file, options_file=options_file)


def create_line_json():
    import json

    line_types = {
        'na': ('narrow', 250.0, (0.0,1200.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
        'br': ('broad', 2500.0, (500.0,15000.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
        'abs': ('absorp', 450.0, (0.1,2500.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
        'out': ('outflow', 100.0, (0.0,800.0), 0.0, (-0.5,0.5), 0.0, (0.0,1.0),),
    }





def main():
    test_single()
    # test_line()
    # test_config()

    # test_muse_single()
    # test_muse_multi()

    # test_nirspec_single()
    # test_nirspec_aperture()
    # test_miri_single()

    # test_random()


if __name__ == '__main__':
    main()

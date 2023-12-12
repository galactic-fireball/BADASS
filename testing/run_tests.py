from multiprocessing import Process
import papermill as pm
import pathlib
import shutil
import sys
import unittest

TESTING_DIR = pathlib.Path(__file__).resolve().parent
OPTIONS_DIR = TESTING_DIR.joinpath('options')
BADASS_DIR = TESTING_DIR.parent
NOTEBOOKS_DIR = BADASS_DIR.joinpath('example_notebooks')
EX_SPEC_DIR = BADASS_DIR.joinpath('example_spectra')

sys.path.insert(0, str(BADASS_DIR))
import badass

OUTPUT_NOTEBOOK = True

def run_notebook(notebook_name):
    notebook = NOTEBOOKS_DIR.joinpath(notebook_name)
    out = notebook_path.with_suffix('.output.ipynb') if OUTPUT_NOTEBOOK else None
    pm.execute_notebook(notebook, out, cwd=NOTEBOOKS_DIR, log_output=True)


LOG_FILES = ['best_model_components.fits', 'log_file.txt', 'par_table.fits']

class NotebookTest(unittest.TestCase):

    # notebooks = ['BADASS3_corner_plot_example.ipynb', 'BADASS3_ifu_MUSE.ipynb', 'BADASS3_multi_spectra_options.ipynb',
        #              'BADASS3_single_spectrum.ipynb', 'BADASS3_ifu_LVS_Rodeo_Cube.ipynb', 'BADASS3_line_test_example.ipynb',
        #              'BADASS3_nonSDSS_single_spectrum.ipynb', 'BADASS3_config_test_example.ipynb', 'BADASS3_ifu_MANGA.ipynb', 'BADASS3_multi_spectra.ipynb']

    def test_metal_absorp(self):
        run_notebook('BADASS3_Metal_Absp.ipynb')

        outdir = EX_SPEC_DIR.joinpath('4-test', 'metal_abs_test')
        self.assertTrue(outdir.exists())

        for file in ['metal_abs_test_bestfit.html', 'input_spectrum.pdf', 'max_likelihood_fit.pdf']:
            self.assertTrue(outdir.joinpath(file).exists())

        for file in LOG_FILES:
            self.assertTrue(outdir.joinpath('log', file).exists())


    def test_auto_corr(self):
        run_notebook('BADASS3_autocorr_example.ipynb')

        outdir = EX_SPEC_DIR.joinpath('3-test', 'autocorr_test')
        self.assertTrue(outdir.exists())

        for file in ['best_fit_model.pdf', 'fitting_region.pdf', 'max_likelihood_fit.pdf']:
            self.assertTrue(outdir.joinpath(file).exists())

        for file in LOG_FILES:
            self.assertTrue(outdir.joinpath('log', file).exists())


    def test_notebooks(self):
        self.skipTest('ignore')

        # notebooks = ['BADASS3_Metal_Absp.ipynb', 'BADASS3_corner_plot_example.ipynb', 'BADASS3_ifu_MUSE.ipynb', 'BADASS3_multi_spectra_options.ipynb',
        #              'BADASS3_single_spectrum.ipynb', 'BADASS3_autocorr_example.ipynb', 'BADASS3_ifu_LVS_Rodeo_Cube.ipynb', 'BADASS3_line_test_example.ipynb',
        #              'BADASS3_nonSDSS_single_spectrum.ipynb', 'BADASS3_config_test_example.ipynb', 'BADASS3_ifu_MANGA.ipynb', 'BADASS3_multi_spectra.ipynb']
        notebooks = ['BADASS3_ifu_MANGA.ipynb', 'BADASS3_multi_spectra.ipynb']

        for notebook in notebooks:
            notebook_path = NOTEBOOKS_DIR.joinpath(notebook)
            out = notebook_path.with_suffix('.output.ipynb') if OUTPUT_NOTEBOOK else None
            pm.execute_notebook(notebook_path, out, cwd=NOTEBOOKS_DIR, log_output=True)


class ScriptTest(unittest.TestCase):
    def test_sdss_multi(self):

        # self.skipTest('ignore')
        options_file = OPTIONS_DIR.joinpath('sdss_single.py')

        tests = [
            EX_SPEC_DIR.joinpath('0-test', 'spec-1087-52930-0084.fits'),
            EX_SPEC_DIR.joinpath('1-test', 'spec-7748-58396-0782.fits'),
            EX_SPEC_DIR.joinpath('2-test', 'spec-2756-54508-0579.fits'),
            EX_SPEC_DIR.joinpath('3-test', 'spec-0266-51602-0151.fits'),
        ]

        run_dirs = [spec.parent.joinpath('result') for spec in tests]
        for outdir in run_dirs:
            if outdir.exists():
                shutil.rmtree(str(outdir))
            outdir.mkdir(parents=True, exist_ok=True)

        badass.run_BADASS(tests, run_dir=run_dirs, options_file=options_file, nprocesses=len(tests))

        for outdir in run_dirs:
            self.assertTrue(outdir.exists())
            for file in ['best_fit_model.pdf', 'input_spectrum.pdf', 'max_likelihood_fit.pdf']:
                self.assertTrue(outdir.joinpath(file).exists())
            for file in LOG_FILES:
                self.assertTrue(outdir.joinpath('log', file).exists())


class TestProcess(Process):
    def __init__(self, test):
        Process.__init__(self)
        self.test = test

    def run(self):
        suite = unittest.TestSuite()
        suite.addTest(self.test)
        unittest.TextTestRunner().run(suite)



def test_regular():
    unittest.main()
    # NotebookTests().test_single()


def test_multiprocess():
    tests = []
    for test_name in ['test_notebooks']:
        tests.append(TestProcess(NotebookTest(test_name)))
    for test_name in ['test_sdss_multi']:
        tests.append(TestProcess(ScriptTest(test_name)))

    for test in tests:
        test.start()
    for test in tests:
        test.join()


# % python -m unittest testing/run_tests.py
if __name__ == '__main__':
    # test_regular()
    test_multiprocess()


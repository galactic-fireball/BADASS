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
    out = notebook.with_suffix('.output.ipynb') if OUTPUT_NOTEBOOK else None
    pm.execute_notebook(notebook, out, cwd=NOTEBOOKS_DIR, log_output=True)


LOG_FILES = ['best_model_components.fits', 'log_file.txt', 'par_table.fits']

class NotebookTest(unittest.TestCase):

    def test_corner_plot(self):
        run_notebook('BADASS3_corner_plot_example.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('3-test', 'corner_plot_test')
        test_files.append(outdir)

        for file in ['best_fit_model.pdf', 'input_spectrum.pdf', 'max_likelihood_fit.pdf', 'corner.pdf']:
            test_files.append(outdir.joinpath(file))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_ifu_muse(self):
        run_notebook('BADASS3_ifu_MUSE.ipynb')

        test_files = []
        muse_dir = EX_SPEC_DIR.joinpath('MUSE')
        subcube_dir = muse_dir.joinpath('NGC1068_subcube')
        test_files.append(subcube_dir)

        for spax in ['3_8', '3_9', '4_8', '4_9']:
            spax_dir = subcube_dir.joinpath('spaxel_%s'%spax)
            test_files.append(spax_dir)
            test_files.append(spax_dir.joinpath('spaxel_%s.fits'%spax))
            for file in LOG_FILES:
                test_files.append(spax_dir.joinpath('MCMC_output_1', 'log', file))

        cube_out_dir = muse_dir.joinpath('MCMC_output_1')
        test_files.append(cube_out_dir.joinpath('best_model_components_plots', 'MODEL.pdf'))
        test_files.append(cube_out_dir.joinpath('partable_plots', 'POWER_AMP.pdf'))
        test_files.append(cube_out_dir.joinpath('log', 'cube_best_model_components.fits'))
        test_files.append(cube_out_dir.joinpath('log', 'cube_par_table.fits'))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_multi_options(self):
        run_notebook('BADASS3_multi_spectra_options.ipynb')

        test_files = []
        for i in range(4):
            outdir = EX_SPEC_DIR.joinpath('%d-test'%i, 'multi_test_opt')
            test_files.append(outdir)
            test_files.append(outdir.joinpath('max_likelihood_fit.pdf'))
            for file in LOG_FILES:
                test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_single(self):
        run_notebook('BADASS3_single_spectrum.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('2-test', 'single_spec_test')
        test_files.append(outdir)
        
        for file in ['2-test_bestfit.html', 'input_spectrum.pdf', 'max_likelihood_fit.pdf']:
            test_files.append(outdir.joinpath(file))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_ifu_rodeo(self):
        run_notebook('BADASS3_ifu_LVS_Rodeo_Cube.ipynb')

        test_files = []
        rodeo_dir = EX_SPEC_DIR.joinpath('LVS_Rodeo_cube')
        subcube_dir = rodeo_dir.joinpath('subcube_1386')
        test_files.append(subcube_dir)

        for spax in ['24_23', '24_24', '25_23', '25_24']:
            spax_dir = subcube_dir.joinpath('spaxel_%s'%spax)
            test_files.append(spax_dir)
            test_files.append(spax_dir.joinpath('spaxel_%s.fits'%spax))
            for file in LOG_FILES:
                test_files.append(spax_dir.joinpath('MCMC_output_1', 'log', file))

        cube_out_dir = rodeo_dir.joinpath('MCMC_output_1')
        test_files.append(cube_out_dir.joinpath('best_model_components_plots', 'MODEL.pdf'))
        test_files.append(cube_out_dir.joinpath('partable_plots', 'POWER_AMP.pdf'))
        test_files.append(cube_out_dir.joinpath('log', 'cube_best_model_components.fits'))
        test_files.append(cube_out_dir.joinpath('log', 'cube_par_table.fits'))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_line_test(self):
        run_notebook('BADASS3_line_test_example.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('2-test', 'line_test')
        test_files.append(outdir)
        test_files.append(outdir.joinpath('2-test_bestfit.html'))
        test_files.append(outdir.joinpath('max_likelihood_fit.pdf'))
        test_files.append(outdir.joinpath('line_test_results', 'test_results.pkl'))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_nonsdss(self):
        run_notebook('BADASS3_nonSDSS_single_spectrum.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('J000338-LRIS-test', 'nonsdss_test')
        test_files.append(outdir)
        test_files.append(outdir.joinpath('J000338-LRIS-test_bestfit.html'))
        test_files.append(outdir.joinpath('max_likelihood_fit.pdf'))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_config_test(self):
        run_notebook('BADASS3_config_test_example.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('2-test', 'config_test')
        test_files.append(outdir)
        
        for file in ['2-test_bestfit.html', 'input_spectrum.pdf', 'max_likelihood_fit.pdf']:
            test_files.append(outdir.joinpath(file))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        test_files.append(outdir.joinpath('config_test_results', 'test_results.pkl'))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_ifu_manga(self):
        run_notebook('BADASS3_ifu_MANGA.ipynb')

        test_files = []
        manga_dir = EX_SPEC_DIR.joinpath('MANGA')
        subcube_dir = manga_dir.joinpath('manga-9485-12705-LOGCUBE')
        test_files.append(manga_dir)

        for spax in ['37_38', '38_38',]:
            spax_dir = subcube_dir.joinpath('spaxel_%s'%spax)
            test_files.append(spax_dir)
            test_files.append(spax_dir.joinpath('spaxel_%s.fits'%spax))
            for file in LOG_FILES:
                test_files.append(spax_dir.joinpath('MCMC_output_1', 'log', file))

        cube_out_dir = manga_dir.joinpath('MCMC_output_1')
        test_files.append(cube_out_dir.joinpath('best_model_components_plots', 'MODEL.pdf'))
        test_files.append(cube_out_dir.joinpath('partable_plots', 'POWER_AMP.pdf'))
        test_files.append(cube_out_dir.joinpath('log', 'cube_best_model_components.fits'))
        test_files.append(cube_out_dir.joinpath('log', 'cube_par_table.fits'))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_multi(self):
        run_notebook('BADASS3_multi_spectra.ipynb')

        test_files = []
        for i in range(4):
            outdir = EX_SPEC_DIR.joinpath('%d-test'%i, 'multi_test')
            test_files.append(outdir)
            test_files.append(outdir.joinpath('max_likelihood_fit.pdf'))
            for file in LOG_FILES:
                test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_metal_absorp(self):
        run_notebook('BADASS3_Metal_Absp.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('4-test', 'metal_abs_test')
        test_files.append(outdir)

        for file in ['4-test_bestfit.html', 'input_spectrum.pdf', 'max_likelihood_fit.pdf']:
            test_files.append(outdir.joinpath(file))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_auto_corr(self):
        run_notebook('BADASS3_autocorr_example.ipynb')

        test_files = []
        outdir = EX_SPEC_DIR.joinpath('3-test', 'autocorr_test')
        test_files.append(outdir)

        for file in ['best_fit_model.pdf', 'input_spectrum.pdf', 'max_likelihood_fit.pdf']:
            test_files.append(outdir.joinpath(file))

        for file in LOG_FILES:
            test_files.append(outdir.joinpath('log', file))

        for file in test_files:
            self.assertTrue(file.exists(), msg=str(file))


    def test_notebooks(self):
        self.skipTest('ignore')

        notebooks = ['BADASS3_Metal_Absp.ipynb', 'BADASS3_corner_plot_example.ipynb', 'BADASS3_ifu_MUSE.ipynb', 'BADASS3_multi_spectra_options.ipynb',
                     'BADASS3_single_spectrum.ipynb', 'BADASS3_autocorr_example.ipynb', 'BADASS3_ifu_LVS_Rodeo_Cube.ipynb', 'BADASS3_line_test_example.ipynb',
                     'BADASS3_nonSDSS_single_spectrum.ipynb', 'BADASS3_config_test_example.ipynb', 'BADASS3_ifu_MANGA.ipynb', 'BADASS3_multi_spectra.ipynb']

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
    test_names = ['test_corner_plot', 'test_ifu_muse', 'test_multi_options', 'test_single', 'test_ifu_rodeo', 'test_line_test',
                  'test_nonsdss', 'test_config_test', 'test_ifu_manga', 'test_multi', 'test_metal_absorp', 'test_auto_corr']

    tests = [TestProcess(NotebookTest(test_name)) for test_name in test_names]
    for test in tests:
        test.start()
    for test in tests:
        test.join()


def clean_tests():
    test_dirs = ['single_spec_test', 'corner_plot_test', 'multi_test', 'multi_test_opt', 'nonsdss_test', 'line_test', 'config_test', 'metal_abs_test', 'autocorr_test', 'MCMC_output*', 'fitting_aperture.pdf', 'result']

    for tdir in test_dirs:
        for d in EX_SPEC_DIR.glob('**/%s'%tdir):
            print(d)
            shutil.rmtree(str(d))

    for note_out in NOTEBOOKS_DIR.glob('*.output.ipynb'):
        print(note_out)
        note_out.unlink()


# % python -m unittest testing/run_tests.py
if __name__ == '__main__':
    # test_regular()
    test_multiprocess()
    # clean_tests()


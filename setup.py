from setuptools import setup


setup(name="ensnest",
      version="0.1-alpha",
      packages=["ensnest"],
      author='djanloo',
      author_email='becuzzigianluca@gmail.com',
      install_requires = [
            'numpy',
            'scipy',
            'tqdm',
            'matplotlib'
      ],
      docs_extras = [
            'Sphinx = 4.2.0',  # Force RTD to use >= 3.0.0
            'docutils',
            'pylons-sphinx-themes >= 1.0.8',  # Ethical Ads
            'pylons_sphinx_latesturl',
            'repoze.sphinx.autointerface',
            'sphinxcontrib-autoprogram',
      ]
    )

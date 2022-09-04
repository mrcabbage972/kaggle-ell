from setuptools import setup, find_packages

setup(name='kaggle_project',
      version='1.0',
      packages=find_packages(),
      entry_points={
          'console_scripts': ['run_solution=kaggle_project.run_solution:main']
      },
    package_data={
      'project': ['config/*.yaml', 'config/env/*.yaml', 'config/model/*.yaml', 'config/workflow/*.yaml'],
     },
      install_requires=[
          'pyyaml==6.0',
          'requests',
          'scikit-learn',
          'scipy',
          'pandas',
          'numpy',
          'hydra-core',
          'tensorboardx',
          'datasets',
          'wandb',
          'tqdm'
      ],
     )
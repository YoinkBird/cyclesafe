from setuptools import setup

setup(name="cyclesafemodel",
        version='0.00001',
        description='CycleSafe: machine learning model for predicting safe bicycle routes',
        author='YoinkBird',
        author_email='yoinkbird2010@gmail.com',
        url='https://github.com/YoinkBird/cyclesafe',
        packages=['modelmanager'],
        license="GNU GPLv3.0: https://www.gnu.org/licenses/gpl-3.0.en.html",
        # py_modules=['modelmanager.model'],
        # crude copypasta from requirements.txt; sin! https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/
        install_requires=[
            'cycler==0.11.0',
            'fonttools==4.38.0',
            'joblib==1.2.0',
            'kiwisolver==1.4.4',
            'matplotlib==3.5.3',
            'numpy==1.17.4',
            'packaging==23.0',
            'pandas==0.23.2',
            'Pillow==9.4.0',
            'pyparsing==3.0.9',
            'python-dateutil==2.8.2',
            'pytz==2022.7',
            'scikit-learn==0.19.2',
            'scipy==1.7.3',
            'seaborn==0.9.0',
            'six==1.16.0',
            'threadpoolctl==3.1.0',
            'typing_extensions==4.4.0',
        ],
        )


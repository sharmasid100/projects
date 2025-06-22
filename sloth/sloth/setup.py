from setuptools import setup, find_packages

setup(
    name='sloth',
    version='0.1.0',
    description='Sloth programming language interpreter',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/slothlang',  # Optional
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'sloth': ['parsetab.py']  # Include generated parser tables
    },
    entry_points={
        'console_scripts': [
            'sloth = sloth.__main__:main',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Interpreters',
    ],
)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# HCCLang - Adapted for Huawei HCCL Collective Communication Library

from setuptools import setup, find_packages

# 读取README文件作为长描述
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "HCCLang - 华为HCCL集合通信算法领域特定语言"

setup(
    name='hcclang',
    version='1.1.0',
    description='华为HCCL集合通信算法领域特定语言',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='HCCLang Development Team',
    author_email='hcclang-dev@example.com',
    url='https://github.com/huawei/hcclang',
    license='MIT',
    
    packages=find_packages(),
    
    # 命令行工具
    entry_points={
        'console_scripts': [
            'hcclang = hcclang.__main__:main',
        ],
    },
    
    # 运行时依赖
    install_requires=[
        'matplotlib>=3.5.0',
        'networkx>=2.8.0',
        'numpy>=1.21.0',
        'igraph>=0.10.0',
        'tabulate>=0.9.0',
        'humanfriendly>=10.0',
        'argcomplete>=2.0.0',
        'lxml>=4.6.0',
        'z3-solver>=4.8.0',
    ],
    
    # 额外依赖（可选安装）
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'jupyter>=1.0.0',
            'jupyterlab>=3.4.0',
        ],
        'visualization': [
            'graphviz>=0.20.0',
            'pandas>=1.3.0',
        ],
        'all': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'jupyter>=1.0.0',
            'jupyterlab>=3.4.0',
            'graphviz>=0.20.0',
            'pandas>=1.3.0',
        ]
    },
    
    # Python版本要求
    python_requires='>=3.8',
    
    # 分类信息
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
    ],
    
    # 关键词
    keywords='hccl collective communication distributed computing machine learning',
    
    # 项目URLs
    project_urls={
        'Documentation': 'https://github.com/huawei/hcclang/docs',
        'Source': 'https://github.com/huawei/hcclang',
        'Tracker': 'https://github.com/huawei/hcclang/issues',
    },
    
    # 包含数据文件
    include_package_data=True,
    package_data={
        'hcclang': [
            'programs/*.py',
            'topologies/*.py',
        ],
    },
    
    # zip安全
    zip_safe=False,
)

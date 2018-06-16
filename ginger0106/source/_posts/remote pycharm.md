---
layout: post
title: "pycharm与远程访问"
date: 2018-6-16 18:40
comments: true
tags: 
	- 技术
---


```
heyhao@ubuntu:~$ jupyter notebook --generate-config
Writing default config to: /home/heyhao/.jupyter/jupyter_notebook_config.py

```

```
heyhao@ubuntu:~$ ipython
Python 3.6.3 |Anaconda custom (64-bit)| (default, Oct 13 2017, 12:02:49) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.1.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from notebook.auth import passwd

In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:48216400b3ae:8062c5ac4a044ec3afaaa15a7ffaf6ce3259c1b0'


```

<https://www.jianshu.com/p/c6697fd9d1bf>
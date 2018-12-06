#! /usr/bin/python
# encoding: utf-8
import yaml
import gzip
import shutil, os
import sys, getopt

path='/root/runtime'
yaml_file = open(path + '/JOBCONFIG.yaml', 'r')
job_config = yaml.load(yaml_file)
yaml_file.close()

job_type = job_config.get('type')
config = job_config.get('job-config')

if job_type == "dl":
    from dl_template import *
    run_model(path, config)
if job_type == "media":
    from media_template import *
    mediaProcess(path, config)
elif job_type == "run_code":
    code = config.get('code')
    os.system(code)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile

import time
import shutil
FNULL = open(os.devnull, 'w')


# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice:
    """
    Main Class to compute the SPICE metric 
    """
    def __init__(self):
        pass

    def float_convert(self, obj):
        try:
          return float(obj)
        except:
          return np.nan

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "test" : hypo[0],
              "refs" : ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
          os.makedirs(temp_dir)

        in_file_path = os.path.join(temp_dir, 'test1.json')
        with open(in_file_path,'w') as in_file:    
          json.dump(input_data,in_file)

        out_file_path = os.path.join(temp_dir, 'tmp_results1.json')

        spice_cmd = ['java','-classpath',cwd, '-jar', 
          '-Xmx8G', os.path.join(cwd,'spice-1.0.jar'), in_file_path,
          '-out',out_file_path,
          '-subset', '-silent'
          
        ]
        out = subprocess.run(spice_cmd, stdout=subprocess.PIPE, stderr=FNULL)#
       

        # Read and process results
        with open(out_file_path) as data_file:    
          results = json.load(data_file)

        scores = []
        for item in results:
          scores.append(item['scores']['All']['f'])

        scores = np.array(scores)
        average_score = np.mean(scores)
        return average_score, scores

    def method(self):
        return "SPICE"

    def __del__(self):
        pass



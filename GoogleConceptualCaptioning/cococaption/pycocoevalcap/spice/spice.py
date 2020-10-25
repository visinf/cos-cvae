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
        #cwd = os.path.dirname(os.path.abspath(__file__))
        #cache_dir=os.path.join(cwd, CACHE_DIR, str(time.time()))
        #self.cache_dir = cache_dir
        #if not os.path.exists(cache_dir):
        #  os.makedirs(cache_dir)
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

        #cwd = os.path.dirname(os.path.abspath(__file__))
        #temp_dir=os.path.join(cwd, TEMP_DIR)
        #if not os.path.exists(temp_dir):
        #  os.makedirs(temp_dir)
        with open('/BS/apr/work/nocaps/image-feature-extractors/test1.json','w') as in_file:    
          json.dump(input_data,in_file)
        #in_file = './test1.json'#tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        #in_file.write(json.dumps(input_data, indent=2).encode('utf-8'))
        #in_file.close()

        # Start job
        #out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        #out_file.close()

       # out = subprocess.run(['java', '-classpath','./SPICE-1.0/','-jar','-Xmx8G',
       #'./SPICE-1.0/spice-1.0.jar','./test1.json','-cache','./spice_cache','-out',
       #'tmp_results1.json','-subset','-silent'], stdout=subprocess.PIPE, stderr=FNULL)#
  

        spice_cmd = ['java','-classpath','/BS/apr/work/nocaps/image-feature-extractors/SPICE-1.0/', '-jar', 
          '-Xmx8G', '/BS/apr/work/nocaps/image-feature-extractors/SPICE-1.0/spice-1.0.jar', '/BS/apr/work/nocaps/image-feature-extractors/test1.json',
          '-out','/BS/apr/work/nocaps/image-feature-extractors/tmp_results1.json',
          '-subset', '-silent'
          
        ]#'-silent' '-cache', self.cache_dir, '-cache','/BS/apr/work/nocaps/image-feature-extractors/spice_cache',
        out = subprocess.run(spice_cmd, stdout=subprocess.PIPE, stderr=FNULL)#
        #    cwd=os.path.dirname(os.path.abspath(__file__)))
        #out = subprocess.run(['java', '-classpath','./SPICE-1.0/','-jar','-Xmx8G','./SPICE-1.0/spice-1.0.jar','./test1.json','-cache','./spice_cache','-out','tmp_results1.json','-subset','-silent'], stdout=subprocess.PIPE, stderr=FNULL)#
        #print(out)

        # Read and process results
        with open('/BS/apr/work/nocaps/image-feature-extractors/tmp_results1.json') as data_file:    
          results = json.load(data_file)
        #os.remove(in_file.name)
        #os.remove(out_file.name)

        scores = []#[None for _ in range(len(results))]
        for item in results:
          scores.append(item['scores']['All']['f'])

        scores = np.array(scores)
        average_score = np.mean(scores)

        '''imgId_to_scores = {}
        spice_scores = []
        for item in results:
          imgId_to_scores[item['image_id']] = item['scores']
          spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
          # Convert none to NaN before saving scores over subcategories
          score_set = {}
          for category,score_tuple in imgId_to_scores[image_id].items():
            score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
          scores.append(score_set)
        print('average_score ',average_score)
        print('scores ',scores)'''
        return average_score, scores

    def method(self):
        return "SPICE"

    def __del__(self):
        pass
        #shutil.rmtree(self.cache_dir)



import pathlib
import re
import random
from time import time

import torch
from transformers import AutoTokenizer, logging

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
import requests
import os
import gradio as gr
import subprocess
import modules.shared

from modules.api.models import *
from modules.api import api
import cudf

def cureTagsV1(tag):
  
  CSV_PATH = pathlib.Path(__file__).parent+"/tagsv3.csv"

  processed_tags = set()

  for t in tag:
      for subtag in t.split(","):
          processed_tags.add(subtag.lower().replace("_", " ").strip())

  gdf = cudf.read_csv(
      CSV_PATH,
      header=None,
      dtype={0: 'str'},  
      names=['col0']
  )

  processed = gdf['col0'].str.lower().str.replace('_', ' ').str.strip()

  matches = processed.isin(list(processed_tags))

  found_processed = processed[matches].unique().to_arrow().to_pylist()

  newtags = []
  for i in range(len(found_processed)):
      found_processed[i] = found_processed[i].lower().replace("_", " ").strip()
  for t in tag:
    newtag = []
    for ftp in t.split(","):
        if ftp.lower().replace("_", " ").strip() not in found_processed:
          newtag.append(ftp)
    newtags.append(",".join(newtag))

  return newtags

def cureTagsV2(tag, keeptags):
  CSV_PATH = pathlib.Path(__file__).parent+"/tagsv2.csv"
  remcategs = ["FACE_CHARACTERISTIC","GENERAL_CHARACTERISTIC","RACE"]
  processed_tags = set()

  for t in tag:
      for subtag in t.split(","):
          processed_tags.add(subtag.lower().replace("_", " ").strip())

  gdf = cudf.read_csv(
      CSV_PATH,
      header=None,
      dtype={0: 'str',1:'str'},  
      names=['col0','col1']
  )

  processed_col1 = gdf['col1'].str.lower().str.replace('_', ' ').str.strip()
  valid_colv1 = gdf['col1'].str.lower().str.replace('_', ' ').str.strip()
  valid_colv2 = ~valid_colv1.isin(valid_colv1)
  valid_col0 = gdf['col0'].isin(remcategs)
  matches = processed_col1.isin(list(processed_tags))  & valid_col0 & valid_colv2

  found_pairs = gdf[matches][['col0', 'col1']].drop_duplicates().to_pandas()

  foundtags = []
  for _, row in found_pairs.iterrows():
    foundtags.append(row['col1'].lower().replace("_", " ").strip())

  newtags = []
  for t in tag:
    newtag = []
    for ftp in t.split(","):
        if ftp.lower().replace("_", " ").strip() not in foundtags:
          newtag.append(ftp)
    newtags.append(",".join(newtag))


  return newtags

def dtg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/mikww/curetags")
    async def dtg(
        tags: list = Body("query", title='tags'),
        keep_tags: list = Body("none", title='keep_tags'),
    ):
      tagst = tags
      taglist = []
      for i,tag in enumerate(tags):
        taglist.append(tag["tags"]["tag"])
      newtags = cureTagsV1(taglist)
      newtags = cureTagsV2(newtags)
      for i,tag in enumerate(tags):
        tagst[i]["tags"]["tag"] = taglist[i]
      return tagst
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(dtg_api)
except:
    pass

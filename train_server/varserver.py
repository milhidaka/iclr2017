#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import json
import base64
import sqlite3
import threading
import bottle
import time
import varserver_config
import varserverapi

thread_local = threading.local()

def get_thread_db():
    if not hasattr(thread_local, 'db'):
        thread_local.db = varserverapi.open_db()
    return thread_local.db

def set_headers(content_type="application/json; charset=utf-8"):
    bottle.response.content_type = content_type
    bottle.response.set_header("Access-Control-Allow-Origin", "*")

def read_meta(blob_id):
    db = get_thread_db()
    cur = db.cursor()
    cur.execute("SELECT id, created_at, size, status, name FROM objs WHERE id = ?", (blob_id, ))
    row = cur.fetchone()
    db.commit()
    if row is not None:
        return {"id":row[0], "created_at":row[1], "size":row[2], "status":row[3], "name":row[4]}
    else:
        raise ValueError("Record for selected id not found")

@bottle.route("/write/<id:int>", method = 'POST')
def write(id):
    blob_id = id
    close = int(bottle.request.query["close"])
    body = bottle.request.body.read()
    with open(varserverapi.make_path(blob_id), "ab") as f:
        f.write(body)
        size = f.tell()

    db = get_thread_db()
    cur = db.cursor()
    obj_status = varserverapi.STATUS_COMPLETE if close else varserverapi.STATUS_WRITING
    cur.execute("UPDATE objs SET size=?, status=? WHERE id=?", (size, obj_status, blob_id))
    db.commit()
    jsonobj = {"id":blob_id}
    jsonstr = json.dumps(jsonobj)
    set_headers()
    return jsonstr

@bottle.route("/write/<id:int>", method = 'OPTIONS')
def write_op(id):
    # CORS
    set_headers()
    bottle.response.set_header("Access-Control-Allow-Headers", "Content-Type")
    return ""

@bottle.route("/stat/<id:int>")
def stat(id):
    set_headers()
    #divide file by varserver_config.RESPONSE_BLOCK_SIZE
    meta = read_meta(id)
    if meta is None or meta["status"] != varserverapi.STATUS_COMPLETE:
        return ""
    total_size = meta["size"]
    blocks = []
    remaining_size = total_size
    while remaining_size > 0:
        body_size = min(remaining_size, varserver_config.RESPONSE_BLOCK_SIZE)
        blocks.append({"body_size":body_size})
        remaining_size -= body_size

    jsonobj = {"id":id, "comment":meta["name"], "size":total_size, "blocks":blocks}
    jsonstr = json.dumps(jsonobj)
    return jsonstr

@bottle.route("/stat/<id:int>", method = 'OPTIONS')
def stat_op(id):
    # CORS
    set_headers()
    bottle.response.set_header("Access-Control-Allow-Headers", "Content-Type")
    return ""

@bottle.route("/read/<id:int>")
def read(id):
    blob_id = id
    block_index = int(bottle.request.query["block_index"])
    offset = block_index * varserver_config.RESPONSE_BLOCK_SIZE
    with open(varserverapi.make_path(blob_id)) as f:
        f.seek(offset)
        body = f.read(varserver_config.RESPONSE_BLOCK_SIZE)

    set_headers("application/octet-stream")
    return body

def run():
    bottle.run(host=varserver_config.HTTP_HOST, port=varserver_config.HTTP_PORT, server='paste')#"pip install paste" to multi-threaded worker

if __name__ == '__main__':
    run()
